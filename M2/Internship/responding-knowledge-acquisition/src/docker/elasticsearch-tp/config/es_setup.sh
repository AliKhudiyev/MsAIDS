curl -X PUT --data @./mappings_settings.json -H "Content-Type: application/json" http://localhost:9200/es_apollo_test
curl -s -H "Content-Type: application/x-ndjson" -XPOST http://localhost:9200/_bulk --data-binary @./sample_apollo_data.ndjson
