import hdfs

from pyarrow import parquet


# os.environ['HADOOP_HOME'] = 'D:/tools/develop/hadoop-2.9.0'
# os.environ['JAVA_HOME'] = 'D:/tools/develop/java/jdk8'
# os.environ['CLASSPATH'] = '$HADOOP_HOME/bin/hdfs classpath --glob'
# os.environ['ARROW_LIBHDFS_DIR'] = '/apps/app/hadoop-3.3.6/lib/native'


class HdfsService:
    def __init__(self, address: str, user: str = None):
        # self.hdfs_conn = fs.HadoopFileSystem(address)
        self.hdfs_conn = hdfs.InsecureClient(url=address, user=user or 'root')

    def read(self, path: str):
        with self.hdfs_conn.read(path) as reader:
            news_data = parquet.read_table(source=path, filesystem=reader)
            return news_data.to_pandas()

    def read_bak(self, path: str):
        news_data = parquet.read_table(source=path, filesystem=self.hdfs_conn)
        return news_data.to_pandas()
