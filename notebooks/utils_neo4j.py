import logging

from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError

class Neo4jApp:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        # Don't forget to close the driver connection when you are finished
        # with it
        self.driver.close()

    def verify_connection(self):
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            logging.error("Connection to Neo4j failed: %s", e)

    def query(self, query, parameters=None):
        try:
            if parameters is not None:
                record = self.driver.execute_query(query, parameters, database_=self.database)
            else:
                record = self.driver.execute_query(query, database_=self.database)
            return record
        except (DriverError, Neo4jError) as exception:
            pass
            # logging.error("%s raised an error: \n%s", query, exception)
            # raise
