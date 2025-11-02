# Hognose Syntax

Hognose is a curly brace syntax language, generally whitespace-insensitive
except for newlines that can terminate statements.
Comments are the classic scripting language `#`.

```
a = 1
b = 2

# or

a = 1;b = 2

```

Everything is an expression. If there is an expression comprised of multiple
expressions, the final expression is the resultant value. Multiple expressions
can be combined into a tuple with ",".

## Block

Blocks combine expressions:

```
a = {
    2 + 1
    3 - 1
} # 4
b = {
    2 + 1
    3 - 2, 2 - 1
} # (1, -1)
```

## Tuples

```
a = (1,)
b = (2, 3, 4)

a[0] # 1
b[2] # 4
```

## Arrays

```
a = [1, 2]
b = [a, 3]
c = [
    a,
    b,
    c,
    d,
    1,
]
d = []

a[1] # 1
b[0, 1] # (a, 3)
c[1..3, 2...4, -1] # ((b,c), (c, d, 1), (1,))
```

## Associative arrays (Dictionaries)

```
a = [
    "cheese": 1,
    "fries": 2
]
b = [a1: 2, a2: 3]
c = [
    b1: "cheese",
    b2: "fries",
    b3: "chicken",
    b4: "fred",
    1: "dude,
]

a["cheese"] # 1
b[a1, a2] # (2, 3)
```

## Ranges

```
1..4 # 1, 2, 3
1...4 # 1, 2, 3, 4
0..4..2 # 0, 2
0...4..2 # 0, 2, 4
```


