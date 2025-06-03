from flask import Flask,render_template,request,Response,redirect,url_for,session,flash
from predict_note import predict_note
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, template_folder='templates')
app.secret_key = "password_integer"
UPLOAD_FOLDER = 'uploads'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = 'adsdsdsd'

model_path = 'saved_models/note_model.pth'
label_mapping_path = 'saved_models/label_mapping.pth'

@app.route('/', methods = ['GET', 'POST'])
def note_prediction():
    note = None
    if request.method == 'POST':
        if 'audio' not in request.files:
             flash('Not a suitable file')
        
        file = request.files['audio']
        if file.filename == '':
            return 'File is not selected'
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        note = predict_note(model_path, label_mapping_path, filepath)
        os.remove(filepath)
        return redirect(url_for("result", note = note))

    return render_template('note_ui.html')

@app.route('/result')
def result():
    note = request.args.get('note')
    return render_template('result.html', note = note)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug = True)

