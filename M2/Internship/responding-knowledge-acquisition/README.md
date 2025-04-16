# Responding - Knowledge Acquisition

Responding is a project whose ultimate goal is to improve the search engine of a company called (traceparts)[traceparts]. There are several stages in the whole project pipeline and **Knowledge Acquisition** is one of them. This gitlab repo contains the source files as well as the documentation needed for knowledge acquisition which also has its own processing pipeline.

## Example

Several python executables could be used for different stages of the pipeline:

```bash
python3 etl.py rdfizer // ETL from TP-ES to RSP-TDB
python3 etl.py rdfloader // ETL from RSP-TDB to RSP-ES

python3 model.py // to create custom spacy language model (usually run once)
python3 rdfizer.py // to generate triples
python3 rdfloader.py // to update database
```

or you can use command-line interface (CLI):

```bash
./cli.sh rdfall
```

## Pipeline

There are several components of the current pipeline each of which is described in the following subsections. To give a brief overview of the developed pipeline, let's go through each stage quickly:

1. **Storing/Loading (customized) language model.** We have built a custom spacy language model for the project and it has to be loaded before anything else in the pipeline.
2. **Processing database entities.** `rdfizer.py` processes database entities(i.e., user queries/texts) in order to create triples in RDF format.
3. **Loading ontology to the database.** `rdfloader.py` loads the obtained triples from the previous stage of the pipeline to the database server.

### Storing/Loading our language model

Since default spaCy langauge models are overhead for the purpose of the project, we have written a python executable that enables us to create and/or load our custom language model. The custom language model is to be fit to just what we need to do with the database entities(texts). The following command will create and save a custom language model on disk under the `language_model` folder.

```bash
python3 model.py
```

At the end of this process there is going to be a folder in the your directory:

```
./language_model/
```

Loading the language model and its essential pipeline components is done by the *rdfizer.py* so you don't have to worry about it for now.

### Processing database entities

An example database response would look like this:

```
response = 
{
    "took": 2,
    "timed_out": false,
    "_shards": {
        "total": 12,
        "successful": 12,
        "skipped": 0,
        "failed": 0
    },
    "hits": {
        "total": {
            "value": 100,
            "relation": "eq"
        },
        "max_score": 1.0,
        "hits": [...]
	}
}
```

Processing such database requires several steps - extracting database entities, loading the language model, generating stottr instance file, compiling the instance file into wottr (rdf). Database entities are extracted sequentially for some number number of threads and processed in a parallel manner by those threads. These 4 steps are executed by `rdfizer.py` which requires either properly set environment variables or `config.ini` file that contains the information shown below:

```
- DATA_PATH		= // path to (csv) database file
- OTTR_LIB_PATH = // path to ottr template library path
- RDF_OUT_DIR 	= // output directory for generated stottr instance file
- LUTRA_PATH 	= // path to the lutra compiler
```

After setting the environment variables or the config file, type on the terminal:

```bash
python3 rdfizer.py
// or
./cli.sh rdfizer
```

At the end of this process, there is going to be the following files in your directory:

```
- ./src/ottr/generated.stottr
- ./src/ottr/output.rdf
```

### Loading ontology to the database

Coming soon...


## Using Jena for everything

Coming soon...

## Testing

Tests are being developed for several components:

- ID regex heuristic
- Triple generation
- Database operations

Go to the `./src/tests/` folder on your terminal.

### rdfizer

```bash
python3 test_rdfizer.py
```

### rdfloader

Coming soon...

