import sqlite3

connection = sqlite3.connect("movies.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS movies(
    TITLE VARCHAR(100), GENRE VARCHAR(50), ReleaseYear INT, Rating REAL)              
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Actors(
    Name VARCHAR(50),BirthYear INT, Nationality VARCHAR(30))         
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS MovieActors(
    MOVIENAME VARCHAR(100), ACTORNAME VARCHAR(50), Role VARCHAR(50))
""")

cursor.execute("INSERT INTO Movies VALUES('Inception', 'Scie-fi',2010, 8.8)")
cursor.execute("INSERT INTO Movies VALUES('Titanic', 'Romance', 1997, 7.9)")
cursor.execute("INSERT INTO Movies VALUES('The Dark Knight', 'Action', 2008, 9.0)")
cursor.execute("INSERT INTO Movies VALUES('Avatar', 'Fantasy', 2009, 7.8)")
cursor.execute("INSERT INTO Movies VALUES('Interstellar', 'Sci-Fi', 2014, 8.6)")


cursor.execute("INSERT INTO Actors VALUES('Leonardo DiCaprio', 1974, 'American')")
cursor.execute("INSERT INTO Actors VALUES('Christian Bale', 1974, 'British')")
cursor.execute("INSERT INTO Actors VALUES('Heath Ledger', 1979, 'Australian')")
cursor.execute("INSERT INTO Actors VALUES('Sam Worthington', 1976, 'Australian')")
cursor.execute("INSERT INTO Actors VALUES('Matthew McConaughey', 1969, 'American')")


cursor.execute("INSERT INTO MovieActors VALUES('Inception', 'Leonardo DiCaprio', 'Dom Cobb')")
cursor.execute("INSERT INTO MovieActors VALUES('Titanic', 'Leonardo DiCaprio', 'Jack Dawson')")
cursor.execute("INSERT INTO MovieActors VALUES('The Dark Knight', 'Christian Bale', 'Bruce Wayne')")
cursor.execute("INSERT INTO MovieActors VALUES('The Dark Knight', 'Heath Ledger', 'Joker')")
cursor.execute("INSERT INTO MovieActors VALUES('Avatar', 'Sam Worthington', 'Jake Sully')")
cursor.execute("INSERT INTO MovieActors VALUES('Interstellar', 'Matthew McConaughey', 'Cooper')")

print("Movies Title:")
for row in cursor.execute("SELECT * FROM Movies"):
    print(row)

print("\nActors Title:")
for row in cursor.execute("SELECT * FROM Actors"):
    print(row)

print("\nMovieActors Title:")
for row in cursor.execute("SELECT * FROM MovieActors"):
    print(row)

connection.commit()
connection.close()