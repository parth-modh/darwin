from typeguard import typechecked
from serve_core.constant.config_constants import CONFIGS_MAP


@typechecked
class Config:
    """
    Config class to get the configuration based on environment
    """

    def __init__(self, env: str):
        self.env = env
        self._config = CONFIGS_MAP[self.env]

    def db_config(self) -> dict:
        mysql_db = self._config["mysql_db"]
        return {
            "host": mysql_db["host"],
            "user": mysql_db["username"],
            "password": mysql_db["password"],
            "database": mysql_db["database"],
            "port": mysql_db["port"],
        }

    @property
    def get_s3_bucket(self):
        return self._config["s3.bucket"]

    @property
    def get_app_layer_url(self):
        return self._config["app-layer-url"]
