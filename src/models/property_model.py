class Property:
    def __init__(self, data):
        self.id = data.get('id')
        self.price = data.get('price')
        self.area = data.get('area')
        self.neighborhood = data.get('neighborhood')
        self.latitude = data.get('latitude')
        self.longitude = data.get('longitude')
        
    def to_dict(self):
        return {
            'id': self.id,
            'price': self.price,
            'area': self.area,
            'neighborhood': self.neighborhood,
            'latitude': self.latitude,
            'longitude': self.longitude
        }