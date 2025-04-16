# What triple to create

## Namespaces

@prefix tp: <https://ontologies.traceparts.com/> .
@prefix gist: <https://ontologies.semanticarts.com/gist/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

## Business Classes

* Class Part Number --> tp:PartNumber
* Class Part Family --> tp:PartFamily
* Class ID --> gist:ID
* Class Text --> gist:Text

create language classes

## Some properties

* tp:hasPartFamily

## Text

URI generation: tp + 'Text' + urlencode(hash(text))

Triples:

```Turtle
tp:Text-URI   a   gist:Text;
            gist:isExpressedIn     tp:Language-URI;
            gist:containedText       "the text"^^xsd:string.
```

## ID

URI generation: tp + 'ID' + urlencode(ID)

Triples:

```Turtle
tp:ID-URI   a   gist:ID;
            gist:uniqueText     "the id text"^^xsd:string;
            [gist:isPartOf       tp:Text-URI.]
```

## Part Number

A part number (PN) is a specific device or Part Family instance. It is not to be confused with the Part Family (PF) which is the generic description of a device and its features. The Part Number has particular values assigned for each corresponding Part Family feature.

Both the PN and PF have an ID, the PN Number and the PF ID respectively.

URI generation: tp + 'PartNumber' + urlencode(PF ID + : + PN Number)

Triples:

```Turtle
tp:PN-URI   a   tp:PartNumber;
            gist:isIdentifiedBy     tp:PN-Number-URI (ID)
            tp:hasPartFamily        tp:PF-URI;
            gist:isDescribedIn      tp:Text-URI (Text).
```

## Part Family

URI generation: tp + 'PartFamily' + urlencode(PF ID)

Triples:

```Turtle
tp:PF-URI   a   tp:PartFamily;
            gist:isIdentifiedBy     tp:PF-ID-URI (ID);
            gist:isDescribedIn      tp:Text-URI (Text).
```
