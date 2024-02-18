import os

import database_connection
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """"""
    # db connection
    mydb = database_connection.SoccerDatabase(
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        port=os.environ["POSTGRES_PORT"],
    )

    sql_queries = os.listdir(cfg["sql"]["tables_creation_queries_path"])

    for filename in sql_queries:
        if filename.split(".")[0] not in mydb.show_tables():
            with open(
                os.path.join(cfg["sql"]["tables_creation_queries_path"], filename), "r"
            ) as f:
                query = f.read()

                mydb.create_table(query=query)

    mydb.close()


if __name__ == "__main__":
    main()
