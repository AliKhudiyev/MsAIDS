docker build ./jena-fuseki/ -t stain/jena-fuseki:4.0.0
docker build ./elasticsearch-tp/ -t es_apollo_test:latest

mkdir ../../data/fuseki
mkdir ../../data/fuseki-data
mkdir ../../data/es-apollo-data

docker-compose up
