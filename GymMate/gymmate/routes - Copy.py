import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request
from gymmate import app, db, bcrypt
from gymmate.forms import RegistrationForm, LoginForm, UpdateAccountForm
from gymmate.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from subprocess import call
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import time



#------------------------------------------File Upload Code-------------------------------------------


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#@app.route('/uploader')
#def upload_form():
    #return render_template('upload.html')


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif, mp4')
            return redirect(request.url)

#--------------------------------------------------------------------------------------------------------------

@app.route("/")
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)



@app.route("/home")
@login_required
def home():
    return render_template('home.html',title='Home')


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/video")
def video():
	return render_template('video.html',title='Videos')


@app.route("/sending")
def sending():
    return render_template('sending.html',title='Sending')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/done")
@login_required

def display():
    return redirect('static/output.avi')


#-----------------------Python Code-------------------------------------------# 



@app.route("/user",methods=['GET','POST'])
@login_required
def returnuser(): 
    #print(current_user.username)
    #print(current_user.email)
    call(["python","gymmate/userid.py"])
    return redirect("http://192.168.43.186:33/", code=302)


@app.route("/client")
@login_required
def client():
    try:
        cap = cv2.VideoCapture(1)
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect(('127.0.0.1', 8083))
        print(current_user.username)
        start=time.time()
        while (time.time()-start)<=120:
            ret,frame = cap.read()
            cv2.imshow('Recording frame',frame)
            cv2.waitKey(1)
            data = pickle.dumps(frame)
            clientsocket.sendall(struct.pack("L", len(data)) + data)
        cap.release()

    except:
        print("Exception ")
    
    return redirect("http://192.168.43.186:33/sending", code=302)



@app.route("/Server")
@login_required
def server():
    call(['python','gymmate/server-video.py'])
    print('Server Running')
    try:
        path1=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\output.mp4"
        path2=r"C:\Users\ragha\Desktop\Full deployement Code\Gym Mate\gymmate\static\output.mp4"
        os.rename(path1,path2)
        shutil.move(path1,path2)
        os.replace(path1,path2)
    except:
        print("An exception occured")

    return redirect("http://192.168.43.186:33/", code=302)