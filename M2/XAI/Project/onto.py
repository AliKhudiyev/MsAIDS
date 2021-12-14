import owlready2 as owl
import numpy as np


onto = owl.get_ontology("http://test.org/onto.owl")

with onto:
    class Grid(owl.Thing):
        pass

    class Cell(Grid):
        pass

    class is_live_at(Cell >> int, owl.FunctionalProperty):
        pass

    class is_neighbour_of(Cell >> Cell):
        pass

    class number_of_live_neighbours(Cell >> int, owl.FunctionalProperty):
        pass


    rule = owl.Imp()
    rule.set_as_rule("""is_live_at(?c, ?t), number_of_live_neighbours(?c, ?n), greaterThan() -> add(?t1, ?t, 1), is_live_at(?c, ?t1)""")
