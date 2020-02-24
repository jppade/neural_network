import connexion

if __name__ == '__main__':
    
    app = connexion.App(__name__, swagger_ui=True)
    app.add_api('./endpoint.yaml', base_path='/', validate_responses=True)
    app.run(host='0.0.0.0',
            port=9091,
            threaded=True)
