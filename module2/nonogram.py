class Segment:

    def __init__(self, x, y, length, direction):
        self.x = x
        self.y = y
        self.length = length
        self.description = "This shape has not been described yet"

    def area(self):
        return self.x * self.y

    def cells(self):
        if(self.direction == "h"):
            return [Cell(self.x + i, self.y, True) for i in range(length)]
        elif(self.direction == "v"):
            return [Cell(self.x, self.y + i, True) for i in range(length)]
        else:
            print "ERROR CREATING SEGMENT"


class Cell:

    def __init__(self, x, y, tag):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return 2 * self.x + 2 * self.y

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x = self.x * scale
        self.y = self.y * scale


class Row:

    def __init__(self, pattern, length):
        self.pattern = pattern
        self.length = length
        self.patterns = []

    def generate_patterns(self, pattern):



























