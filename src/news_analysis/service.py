from pyarrow import parquet, fs


class HdfsService:
    def __init__(self, host: str, port: int, user: str):
        self.hdfs_conn = fs.HadoopFileSystem(host=host, port=port, user=user or 'root')

    def read(self, path: str):
        news_data = parquet.read_table(source=path, filesystem=self.hdfs_conn)
        return news_data.to_pandas()
