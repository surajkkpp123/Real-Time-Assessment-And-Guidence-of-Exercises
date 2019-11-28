from flask import Flask, request, redirect
from subprocess import call

app = Flask(__name__)

@app.route("/")

def hello_world():
    return "Test 123 "


#---------------------------------Links for Processing----------------------------------#

@app.route("/r_bicep_curl")

def rcall():
	call(["python","r_bicep_curl.py"])
	print('Video Started Processing')
	return redirect("http://192.168.43.186:33/", code=302)


@app.route("/push_ups")

def pcall():
	call(["python","push_ups.py"])
	print('Video started processing')
	return redirect("http://192.168.43.186:33/", code=302)

@app.route("/high_plank")

def hcall():
	call(["python","high_plank.py"])
	print('Video started processing')
	return redirect("http://192.168.43.186:33/", code=302)

@app.route("/leg_raise")
def lcall():
	call(["python","leg_raise.py"])
	
	return redirect("http://192.168.43.186:33/", code=302)

@app.route("/pull_ups")
def pucall():
	call(["python","pull_ups.py"])
	return redirect("http://192.168.43.186:33/", code=302)

@app.route("/deep_squat")
def dcall():
	call(["python","deep_squat.py"])
	return redirect("http://192.168.43.186:33/", code=302)



#-------------------------------------for Communication------------------------------------#


@app.route("/Server")

def server():
	print("Server started listening")
	call(["python","server-video.py"])
	return redirect("http://192.168.43.186:33/", code=302)




@app.route("/Client")

def client():
	print("Client started Sending")
	call(["python","clientcv.py"])
	return redirect("http://192.168.43.186:33/", code=302)




@app.route("/response")

def response():
	return redirect("http://192.168.43.186:33/", code=302)
	

if __name__ == "__main__":
    app.run(host="192.168.43.186",port=8000)
    #app.run(host='0.0.0.0',port=33)