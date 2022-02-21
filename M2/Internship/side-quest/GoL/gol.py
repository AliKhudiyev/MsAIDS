import numpy as np
from owlready2 import *


onto = get_ontology("http://test.org/gol.owl")

with onto:
    class Cell(Thing):
        pass

    class is_on_right_of():
        domain  = [Cell]
        range   = [Cell]

    class is_on_left_of(Cell >> Cell):
        pass

    class is_on_top_of(Cell >> Cell):
        pass

    class is_on_bottom_of(Cell >> Cell):
        pass

    class is_neighbour_of(Cell >> Cell):
        pass

    class is_live_at(Cell >> int):
        pass

    class has_sufficient_neighbours_at(Cell >> int):
        pass

    class has_insufficient_neighbours_at(Cell >> int):
        pass

