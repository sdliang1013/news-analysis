from typing import List

import happybase

from news_analysis import schemas


# import phoenixdb


# os.environ['HADOOP_HOME'] = 'D:/tools/develop/hadoop-2.9.0'
# os.environ['JAVA_HOME'] = 'D:/tools/develop/java/jdk8'
# os.environ['CLASSPATH'] = '$HADOOP_HOME/bin/hdfs classpath --glob'
# os.environ['ARROW_LIBHDFS_DIR'] = '/apps/app/hadoop-3.3.6/lib/native'

# class NewsService:
#
#     def __init__(self, db_url: str):
#         self.conn = phoenixdb.connect(url=db_url, autocommit=True)
#
#     def read(self) -> List[schemas.News]:
#         with self.conn.cursor() as cursor:
#             cursor.execute("SELECT * FROM news")
#             rows = cursor.fetchall()
#         return list(map(lambda x: self._to_news_(x), rows))
#
#     def _to_news_(self, row) -> schemas.News:
#         return schemas.News(**row)


class NewsService:

    def __init__(self, host: str, port: int = None):
        self.conn = happybase.Connection(host=host, port=port or 9090, autoconnect=False)
        self.tab = self.conn.table(schemas.News.table_name())

    def read_news(self) -> List[schemas.News]:
        """
        查询新闻
        :return:
        """
        rows = []
        try:
            self.conn.open()
            for key, data in self.tab.scan():
                rows.append(schemas.News.of(row=data, rid=key.decode()))
            return rows
        finally:
            self.conn.close()

    def analysis_news(self, news_list: List[schemas.News]) -> List[schemas.NewsMetrics]:
        """
        分析新闻,生成指标
        :param news_list:
        :return:
        """
        ...

    def save_metrics(self, metric_list: List[schemas.NewsMetrics]):
        """
        保存指标
        :param metric_list:
        :return:
        """
        ...
