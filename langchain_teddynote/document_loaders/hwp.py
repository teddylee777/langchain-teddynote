from typing import Any, Dict, List, Optional, Iterator
import olefile
import zlib
import struct
import re
import unicodedata
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader


class HWPLoader(BaseLoader):
    """HWP 파일 읽기 클래스. HWP 파일의 내용을 읽습니다."""

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.extra_info = {"source": file_path}
        self._initialize_constants()

    def _initialize_constants(self) -> None:
        """상수 초기화 메서드"""
        self.FILE_HEADER_SECTION = "FileHeader"
        self.HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self.SECTION_NAME_LENGTH = len("Section")
        self.BODYTEXT_SECTION = "BodyText"
        self.HWP_TEXT_TAGS = [67]

    def lazy_load(self) -> Iterator[Document]:
        """HWP 파일에서 데이터를 로드하고 표를 추출합니다.

        Yields:
            Document: 추출된 문서
        """
        load_file = olefile.OleFileIO(self.file_path)
        file_dir = load_file.listdir()

        if not self._is_valid_hwp(file_dir):
            raise ValueError("유효하지 않은 HWP 파일입니다.")

        result_text = self._extract_text(load_file, file_dir)
        yield self._create_document(text=result_text, extra_info=self.extra_info)

    def _is_valid_hwp(self, dirs: List[List[str]]) -> bool:
        """HWP 파일의 유효성을 검사합니다."""
        return [self.FILE_HEADER_SECTION] in dirs and [self.HWP_SUMMARY_SECTION] in dirs

    def _get_body_sections(self, dirs: List[List[str]]) -> List[str]:
        """본문 섹션 목록을 반환합니다."""
        section_numbers = [
            int(d[1][self.SECTION_NAME_LENGTH :])
            for d in dirs
            if d[0] == self.BODYTEXT_SECTION
        ]
        return [
            f"{self.BODYTEXT_SECTION}/Section{num}" for num in sorted(section_numbers)
        ]

    def _create_document(
        self, text: str, extra_info: Optional[Dict] = None
    ) -> Document:
        """문서 객체를 생성합니다."""
        return Document(page_content=text, metadata=extra_info or {})

    def _extract_text(
        self, load_file: olefile.OleFileIO, file_dir: List[List[str]]
    ) -> str:
        """모든 섹션에서 텍스트를 추출합니다."""
        sections = self._get_body_sections(file_dir)
        return "\n".join(
            self._get_text_from_section(load_file, section) for section in sections
        )

    def _is_compressed(self, load_file: olefile.OleFileIO) -> bool:
        """파일이 압축되었는지 확인합니다."""
        with load_file.openstream(self.FILE_HEADER_SECTION) as header:
            header_data = header.read()
            return bool(header_data[36] & 1)

    def _get_text_from_section(self, load_file: olefile.OleFileIO, section: str) -> str:
        """특정 섹션에서 텍스트를 추출합니다."""
        with load_file.openstream(section) as bodytext:
            data = bodytext.read()

        unpacked_data = (
            zlib.decompress(data, -15) if self._is_compressed(load_file) else data
        )

        text = []
        i = 0
        while i < len(unpacked_data):
            header, rec_type, rec_len = self._parse_record_header(
                unpacked_data[i : i + 4]
            )
            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                text.append(rec_data.decode("utf-16"))
            i += 4 + rec_len

        text = "\n".join(text)
        text = self.remove_chinese_characters(text)
        text = self.remove_control_characters(text)
        return text

    @staticmethod
    def remove_chinese_characters(s: str):
        """중국어 문자를 제거합니다."""
        return re.sub(r"[\u4e00-\u9fff]+", "", s)

    @staticmethod
    def remove_control_characters(s):
        """깨지는 문자 제거"""
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    @staticmethod
    def _parse_record_header(header_bytes: bytes) -> tuple:
        """레코드 헤더를 파싱합니다."""
        header = struct.unpack_from("<I", header_bytes)[0]
        rec_type = header & 0x3FF
        rec_len = (header >> 20) & 0xFFF
        return header, rec_type, rec_len
