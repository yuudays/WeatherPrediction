queryCreate="""CREATE TABLE weather (
                                    date DATE NOT NULL,
                                    precipitation FLOAT NOT NULL,
                                    temp_max FLOAT NOT NULL,
                                    temp_min FLOAT NOT NULL,
                                    wind FLOAT NOT NULL,
                                    weather VARCHAR(255) NOT NULL);"""
queryDrop = """DROP TABLE weather"""

queryInsertTable = """
                INSERT INTO weather (date, precipitation, temp_max, temp_min, wind, weather)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
queryInsert = """
                INSERT INTO weather (date, precipitation, temp_max, temp_min, wind, weather)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s);
                """