class PF32_Footer:

    NO_OF_FIELDS = 4

    def __init__(self, frame_number, x, y, z):
        self.frame_number = frame_number
        self.x = x
        self.y = y
        self.z = z
    
    def get_frame_number(self):
        return self.frame_number

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z
    
    
