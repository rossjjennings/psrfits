from textwrap import dedent

class Source:
    def __init__(self, name, model, polyco=None, predictor=None):
        self.name = name
        self.model = model
        self.polyco = polyco
        self.predictor = predictor
    
    @classmethod
    def from_hdulist(cls, hdulist):
        name = hdulist['primary'].header['src_name']
        model = '\n'.join(line[0] for line in hdulist['psrparam'].data)
        polyco = None
        predictor = None
        if 'polyco' in hdulist:
            polyco = hdulist['polyco'].data
        if 't2predict' in hdulist:
            predictor = '\n'.join(line[0] for line in hdulist['t2predict'].data)
        return cls(name, model, polyco, predictor)
    
    def __str__(self):
        return f' {self.name} [...]'
    
    def __repr__(self):
        description = "<xpsrfits.Source>\n"
        included_names = ["name", "model"]
        if self.polyco is not None:
            included_names.append("polyco")
        if self.predictor is not None:
            included_names.append("predictor")
        max_len = max(len(name) for name in included_names)
        for name in included_names:
            key = f"{name}:"
            description += f"    {key:<{max_len + 2}}{self._fmt_prop(name)}\n"
        return description
    
    def _fmt_prop(self, name):
        prop_str = str(getattr(self, name))
        lines = prop_str.split('\n')
        if len(lines) > 1:
            return f'{lines[0]} [...]'
        else:
            return prop_str
