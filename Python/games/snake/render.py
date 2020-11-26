class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.width = 500
        self.height = 500
        self.objects = []
    
    def submit(self, objects):
        self.objects = objects

    def flush(self):
        self.canvas.delete('all')
        for object_ in self.objects:
            if object_.type == 0:   # snake
                for i, block in enumerate(object_.blocks):
                    color = 'green'
                    if i == 0:
                        color = 'orange'
                    
                    x0 = block.x - block.size / 2
                    y0 = block.y - block.size / 2
                    x1 = block.x + block.size / 2
                    y1 = block.y + block.size / 2
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='black')
            else:                   # fruit
                x0 = object_.x - object_.size / 2
                y0 = object_.y - object_.size / 2
                x1 = object_.x + object_.size / 2
                y1 = object_.y + object_.size / 2
                self.canvas.create_oval(x0, y0, x1, y1, fill='red', outline='red')
