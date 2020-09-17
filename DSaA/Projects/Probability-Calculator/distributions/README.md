# Distribution CSV Format

If probability function has only 1 argument:

|x |0 |1 |
|--|--|--|
|? |? |? |
|? |? |? |

Or

|x |0 |1 |
|--|--|--|
|2 |? |? |
|3 |? |? |

If probability function has 2 arguments:

|x |y |0 |1 |
|--|--|--|--|
|0 |  |? |? |
|1 |  |? |? |

If probability function has 3 arguments:

|x |  |0 |1 |
|--|--|--|--|
|y |z |  |  |
|0 |0 |? |? |
|  |1 |? |? |
|1 |0 |? |? |
|  |1 |? |? |

If probability function has 4 arguments:

|a |b |  |0 |1 |
|--|--|--|--|--|
|  |c |d |  |  |
|0 |0 |0 |? |? |
|  |  |1 |? |? |
|  |1 |0 |? |? |
|  |  |1 |? |? |
|1 |0 |0 |? |? |
|  |  |1 |? |? |
|  |1 |0 |? |? |
|  |  |1 |? |? |
