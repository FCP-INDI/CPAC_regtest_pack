from flask import Flask
from summary_app.config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    from summary_app.correlations.main import correlations
    from summary_app.main.routes import main
    from summary_app.errors.handlers import errors
    from summary_app.entity.utils import entity
    from summary_app.logging.logger import logger
    app.register_blueprint(correlations)
    app.register_blueprint(main)
    app.register_blueprint(errors)
    app.register_blueprint(entity)
    app.register_blueprint(logger)

    return app