import feedparser
from urllib.parse import quote
from typing import List, Dict, Optional


class GoogleNews:
    """
    구글 뉴스를 검색하고 결과를 반환하는 클래스입니다.
    """

    def __init__(self):
        """
        GoogleNews 클래스를 초기화합니다.
        base_url 속성을 설정합니다.
        """
        self.base_url = "https://news.google.com/rss"

    def _fetch_news(self, url: str, k: int = 3) -> List[Dict[str, str]]:
        """
        주어진 URL에서 뉴스를 가져옵니다.

        Args:
            url (str): 뉴스를 가져올 URL
            k (int): 가져올 뉴스의 최대 개수 (기본값: 3)

        Returns:
            List[Dict[str, str]]: 뉴스 제목과 링크를 포함한 딕셔너리 리스트
        """
        news_data = feedparser.parse(url)
        return [
            {"title": entry.title, "link": entry.link}
            for entry in news_data.entries[:k]
        ]

    def _collect_news(self, news_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        뉴스 리스트를 정리하여 반환합니다.

        Args:
            news_list (List[Dict[str, str]]): 뉴스 정보를 포함한 딕셔너리 리스트

        Returns:
            List[Dict[str, str]]: URL과 내용을 포함한 딕셔너리 리스트
        """
        if not news_list:
            print("해당 키워드의 뉴스가 없습니다.")
            return []

        result = []
        for news in news_list:
            result.append({"url": news["link"], "content": news["title"]})

        return result

    def search_latest(self, k: int = 3) -> List[Dict[str, str]]:
        """
        최신 뉴스를 검색합니다.

        Args:
            k (int): 검색할 뉴스의 최대 개수 (기본값: 3)

        Returns:
            List[Dict[str, str]]: URL과 내용을 포함한 딕셔너리 리스트
        """
        url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)

    def search_by_keyword(
        self, keyword: Optional[str] = None, k: int = 3
    ) -> List[Dict[str, str]]:
        """
        키워드로 뉴스를 검색합니다.

        Args:
            keyword (Optional[str]): 검색할 키워드 (기본값: None)
            k (int): 검색할 뉴스의 최대 개수 (기본값: 3)

        Returns:
            List[Dict[str, str]]: URL과 내용을 포함한 딕셔너리 리스트
        """
        if keyword:
            encoded_keyword = quote(keyword)
            url = f"{self.base_url}/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
        else:
            url = f"{self.base_url}?hl=ko&gl=KR&ceid=KR:ko"
        news_list = self._fetch_news(url, k)
        return self._collect_news(news_list)
