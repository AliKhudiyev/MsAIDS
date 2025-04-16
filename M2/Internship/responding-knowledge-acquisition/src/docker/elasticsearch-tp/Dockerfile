FROM docker.elastic.co/elasticsearch/elasticsearch:8.1.2

WORKDIR /usr/share/elasticsearch/bin

RUN ./elasticsearch-plugin install analysis-icu
RUN ./elasticsearch-plugin install analysis-kuromoji
RUN ./elasticsearch-plugin install analysis-smartcn
RUN ./elasticsearch-plugin install analysis-stempel
RUN ./elasticsearch-plugin install mapper-size

COPY --chown=elasticsearch:elasticsearch ./config/elasticsearch.yml /usr/share/elasticsearch/config/

RUN mkdir /usr/share/elasticsearch/es_apollo_test_setup/

WORKDIR /usr/share/elasticsearch/es_apollo_test_setup/

COPY --chown=elasticsearch:elasticsearch ./config/mappings_settings.json .
COPY --chown=elasticsearch:elasticsearch ./config/es_setup.sh .
COPY --chown=elasticsearch:elasticsearch ./config/sample_apollo_data.ndjson .