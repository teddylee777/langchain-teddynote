import requests
import json
import time
from typing import Literal


class SynapsoftDocuAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://da.synap.co.kr/api"
        self.headers = {"Content-Type": "application/json"}
        self.fid = None
        self.total_pages = None
        self.max_retries = 1000
        self.delay = 3

    def upload_file(self, file_path):
        """파일을 업로드하고 분석 요청"""
        url = f"{self.base_url}/da"

        files = {"file": open(file_path, "rb")}
        data = {"api_key": self.api_key, "type": "upload"}

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            self.fid = response_data.get("result", {}).get("fid")
            self.total_pages = response_data.get("result", {}).get("total_pages")
            print(f"파일 업로드 성공: {self.fid}")
            print(f"총 페이지 수: {self.total_pages}")
            return self.fid
        else:
            print(f"파일 업로드 실패: {response.status_code}")
            print(response.text)
            return None

    def check_file_status(self):
        """파일 처리 상태 확인"""
        if not self.fid:
            print("파일 ID가 없습니다. 먼저 파일을 업로드하세요.")
            return None

        url = f"{self.base_url}/filestatus/{self.fid}"
        data = {"api_key": self.api_key}

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            file_status = response_data.get("result", {}).get("filestatus")
            return file_status
        else:
            print(f"상태 확인 실패: {response.status_code}")
            print(response.text)
            return None

    def validation_check(self):
        """API 상태 확인"""
        url = f"{self.base_url}/monitor"
        data = {"api_key": self.api_key}

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code == 200:
            print(f"API 상태: {response.status_code}")
            return True
        else:
            print(f"API 상태 확인 실패: {response.status_code}")
            print(response.text)
            return False

    def _get_result(self, page_index=0, result_type="md"):
        """분석 결과 가져오기"""
        if not self.fid:
            print("파일 ID가 없습니다. 먼저 파일을 업로드하세요.")
            return None

        url = f"{self.base_url}/result/{self.fid}"
        data = {"api_key": self.api_key, "page_index": page_index, "type": result_type}

        response = requests.post(url, headers=self.headers, json=data)

        if response.status_code == 200:
            return response.content.decode("utf-8")
        else:
            print(f"결과 가져오기 실패: {response.status_code}")
            print(response.text)
            return None

    def _process_file(
        self, file_path, result_type: Literal["md", "xml", "json"] = "md"
    ):
        """파일 업로드부터 결과 반환까지 전체 프로세스 실행"""
        # 1. 파일 업로드
        if not self.upload_file(file_path):
            return None

        # 2. 파일 상태 확인 (SUCCESS가 될 때까지)
        retries = 0
        while retries < self.max_retries:
            status = self.check_file_status()
            print(f"현재 상태: {status}")

            if status == "SUCCESS":
                break

            retries += 1
            time.sleep(self.delay)

        if retries >= self.max_retries:
            print("최대 재시도 횟수 초과: 파일 처리가 완료되지 않았습니다.")
            return None

        # 3. 모든 페이지의 결과 반환
        all_results = []
        if self.total_pages:
            for page_index in range(self.total_pages):
                print(f"페이지 {page_index+1}/{self.total_pages} 결과 가져오는 중...")
                page_result = self._get_result(
                    page_index=page_index, result_type=result_type
                )
                if page_result:
                    all_results.append(page_result)
                else:
                    print(f"페이지 {page_index} 결과를 가져오지 못했습니다.")

            return all_results
        else:
            # total_pages가 없는 경우 단일 페이지 결과만 반환
            return [self._get_result(result_type=result_type)]
        
    def convert_to_markdown(self, file_path):
        result = self._process_file(file_path, result_type="md")
        return result
    
    def convert_to_xml(self, file_path):
        result = self._process_file(file_path, result_type="xml")
        return result
    
    def convert_to_json(self, file_path):
        result = self._process_file(file_path, result_type="json")
        return result
