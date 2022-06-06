class BBox:
    def __init__(self, textData: str):
        data = textData.split()
        if len(data) != 4:
            raise IndexError("Wrong number of arguments!")
        self.x = int(float(data[0]))
        self.y = int(float(data[1]))
        self.width = int(float(data[2]))
        self.height = int(float(data[3]))

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height
