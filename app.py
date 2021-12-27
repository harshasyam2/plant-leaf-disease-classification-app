from flask import Flask, render_template, request
import os
import cifar10maize

app= Flask(__name__)
app.config["UPLOAD_FOLDER"]="./Uploaded File"

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        leaf_name=request.form["leaf_type"]
        return render_template("prediction.html",leaf_name=leaf_name)

@app.route("/submit",methods=["POST"])
def submit():
    if request.method=="POST":
        input_image=request.files["inp_img"]
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], input_image.filename)
        input_image.save(path)

        image_name=input_image.filename

        image_path=path.split('\\')
        image_path=image_path[0]+"/"+image_name
        pred_acc=cifar10maize.calc_maize_leaf(image_path)
        return render_template("submit.html",accuracy=pred_acc)
    return render_template("submit.html")

if __name__=="__main__":
    app.run(debug=True)