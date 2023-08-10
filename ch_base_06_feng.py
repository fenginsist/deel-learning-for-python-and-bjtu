def test_yield():
    x = 1
    y = 10
    while x < y:
        yield  x
        x += 1

example = test_yield()
# example