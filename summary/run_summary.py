from summary_app import create_app

app = create_app()
app.app_context().push()
#data_path = app.config['DATA_DIR']

if __name__ == "__main__":
    app.run(debug=True)