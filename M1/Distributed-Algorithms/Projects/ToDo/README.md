# Interactive To-Do Web Application

## How to deploy

To start the application:

```bash
docker start postgres-todo && docker start app-todo
```

To stop the application:

```bash
docker stop app-todo && docker stop postgres-todo
```

## Steps to build the project from scratch

1. Install docker
2. Install postgres image for docker
3. Create a container to deploy PostgreSQL database

   ```bash
   docker run --name postgres-todo -e POSTGRES_PASSWORD=Pass123! -p 5432:5432 -d postgres
   ```

4. Create a database and necessary tables for the Spring Boot application

   ```bash
   docker cp [path/to/ToDo/db.psql] postgres-todo:/db.psql
   docker exec -it postgres-todo /bin/sh
   psql -U postgres
   createdb todo_app -h localhost -p 5432 -U postgres
   psql -U postgres < db.psql
   ```

5. Go to the Spring Boot application directory (path/to/ToDo)

   ```bash
   mvn clean install
   mkdir Docker
   echo '
   FROM openjdk:latest
   COPY ToDo-0.0.1-SNAPSHOT.jar /app.jar
   EXPOSE 8080
   CMD ["java", "-jar", "/app.jar"] 
   ' > Dockerfile
   cp target/ToDo-0.0.1-SNAPSHOT.jar Docker
   docker build -t rest-todo Docker
   ```

6. Create and run the container for the *Interactive To-Do web application*

   ```bash
   docker run --name app-todo --link postgres-todo:postgres -p 8080:8080 -d rest-todo
   ```

*Note:* For using CLI of the web application you can type `docker exec -it app-todo /bin/sh`.
