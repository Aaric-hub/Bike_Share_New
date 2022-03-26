from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from trainingModel import trainModel,trainValidation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction, prediction_validation

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=["GET","POST"])
@cross_origin()
def predictRouteClient():
    try:
        if request.method == 'POST' and request.files:
            file = request.files['File']
            file_name = file.filename
            print(file_name)
            file.save(f"Recived_file/{file_name}")
            path = f"Recived_file/{file_name}"
            pred_val = prediction_validation(path=path)
            print("pred_val done")
            pred = prediction(path=path)
            print("prediction done")
            #path,predictions = pred.predictionFromModel()
            return render_template('index.html',prediction_output=f"Prediction File Created \n Predictions are :: {preds}")
        elif request.method == 'POST' and request.files:
            file = request.files['File']
            file_name = file.filename
            file.save(f"Recived_file/{file_name}")
            path = f"Recived_file/{file_name}"
            pred_val = prediction_validation(path)
            
            pred = prediction(path)

            #path,predictions = pred.predictionFromModel()
            return render_template('index.html',prediction_output=f"Prediction File Created \n Predictions are :: {pred}")
        else:
            print('Nothing Matched..')

    except ValueError:
        return render_template('index.html', error_pred = "Error Occurred! %s" %ValueError)
    except KeyError:
        return render_template('index.html', error_pred = "Error Occurred! %s" %KeyError)
    except Exception as e:
        return render_template('index.html', error_pred = "Error Occurred! %s" %e)

@app.route("/train", methods = ["GET","POST"])
@cross_origin()

def trainRouteClient():
    try:
        folder_path = 'Training_File/day.csv'
        if folder_path is not None:
            path = folder_path
            train_val = trainValidation(path)
            

            train_model = trainModel()
            train_model.trainingModel()
            return render_template('index.html',output="Model Saved Successfully at models/initial_model!!!")

    except ValueError:

        return render_template('index.html', error_train = "Error Occurred! %s" %ValueError)

    except KeyError:

        return render_template('index.html', error_train = "Error Occurred! %s" % KeyError)

    except Exception as e:

        return render_template('index.html', error_train = f"Error Occurred! {e}")
    #return Response("Training successful...")
#port = int(os.getenv("PORT",5000))

if __name__ == "__main__":
    #host = '0.0.0.0'
    #port = 5000
    #httpd = simple_server.make_server(host, port, app)
    #print("Serving on %s %d" % (host, port))
    app.run(debug=True)
    #httpd.serve_forever()