
from flask import Flask

app = Flask(__name__)

# Generate secret key = import os >>> os.urandom(24)
app.config['SECRET_KEY'] = '\x8eG\xeb\x0c"g\x86P&\xe8zG\xafwk\x1c\xfa\xd5\xd8\x9b"\xce\x1dr'

from .main import main as main_blueprint

# blueprints for non-auth parts of app
app.register_blueprint(main_blueprint)
