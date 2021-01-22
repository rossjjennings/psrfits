from textwrap import dedent

class Telescope:
    __slots__ = 'name', 'ant_x', 'ant_y', 'ant_z'
    
    def __init__(self, name, ant_x, ant_y, ant_z):
        self.name = name
        self.ant_x = ant_x
        self.ant_y = ant_y
        self.ant_z = ant_z
    
    @classmethod
    def from_header(cls, header):
        return cls(
            header['telescop'],
            header['ant_x'],
            header['ant_y'],
            header['ant_z'],
        )
    
    def __str__(self):
        return f'{self.name} [...]'
    
    def __repr__(self):
        description = f'''
        <xpsrfits.Telescope>
            name = {self.name}
            ant_x = {self.ant_x}
            ant_y = {self.ant_y}
            ant_z = {self.ant_z}
        '''
        return dedent(description)[1:]
