"""
Helper: generate the offline @font-face CSS block with base64-embedded woff2 fonts.
Run once to get the string to paste into main.py.
"""
import base64, os

HERE = os.path.join(os.path.dirname(__file__), '..', 'static', 'fonts')

def b64(name):
    path = os.path.join(HERE, name)
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

orbitron = b64('Orbitron.woff2')
inter300  = b64('Inter-300.woff2')
inter400  = b64('Inter-400.woff2')
inter500  = b64('Inter-500.woff2')
inter600  = b64('Inter-600.woff2')

css = f"""@font-face {{
  font-family: 'Orbitron';
  font-style: normal;
  font-weight: 400 900;
  font-display: swap;
  src: url('data:font/woff2;base64,{orbitron}') format('woff2');
}}
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 300;
  font-display: swap;
  src: url('data:font/woff2;base64,{inter300}') format('woff2');
}}
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('data:font/woff2;base64,{inter400}') format('woff2');
}}
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url('data:font/woff2;base64,{inter500}') format('woff2');
}}
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url('data:font/woff2;base64,{inter600}') format('woff2');
}}"""

print(css)
