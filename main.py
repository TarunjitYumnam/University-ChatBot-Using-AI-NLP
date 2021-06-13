from flask import Flask, render_template, request, redirect, session, jsonify, flash, url_for
import mysql.connector
import os
import requests
import text_process

app = Flask(__name__)
app.secret_key = os.urandom(24)
botname = 'Yumnam_Jr'

conn = mysql.connector.connect(host="localhost", user="root", password="", database="user_profile")
cursor = conn.cursor()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/home')
def home():
    if 'name' in session:
        return render_template('index.html', botname=botname, **locals())
    else:
        return redirect('/')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/login_validation', methods = ['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,password))
    users = cursor.fetchall()
    print(users)
    # return 'Yes'
    if len(users)>0:
        session['name'] = users[0][0]
        # act = session['name']
        print(session['name'])
        return redirect('/home')
    else:
        return redirect('/')

@app.route('/admin_validation', methods = ['POST'])
def admin_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""SELECT * FROM `admin_table` WHERE `email` LIKE '{}' AND `password` LIKE '{}'""".format(email,
                                                                                                             password))
    admin_login = cursor.fetchall()
    if len(admin_login) > 0:
        session['email'] = admin_login[0][0]
        return redirect(url_for('Index'))
    else:
        return redirect('/admin')

@app.route('/table')
def Index():
    cursor.execute("""SELECT  * FROM `user`""")
    data = cursor.fetchall()

    return render_template('adminDash.html', users=data )



@app.route('/insert', methods = ['POST'])
def insert():

    if request.method == "POST":
        flash("Data Inserted Successfully")
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        cursor.execute("""INSERT INTO `user` (`name`,`email`,`password`) VALUES
            ('{}', '{}', '{}')""".format(name, email, password))
        conn.commit()
        return redirect(url_for('Index'))

@app.route('/delete/<name>', methods = ['GET', 'POST'])
def delete(name):
    flash("Record Has Been Deleted Successfully")
    cursor.execute("""DELETE FROM `user` WHERE name = %s""", (name,))
    conn.commit()
    return redirect(url_for('Index'))

@app.route('/update',methods=['POST','GET'])
def update():

    if request.method == 'POST':
        name = request.form['name']
        #name = name.replace("%20"," ")
        email = request.form['email']
        password = request.form['password']
        cursor.execute("""
               UPDATE `user`
               SET email=%s, password=%s
               WHERE  name=%s
            """, (email, password, name))
        flash("Data Updated Successfully")
        conn.commit()
        return redirect(url_for('Index'))


@app.route('/add_user', methods = ['POST'])
def add_user():
    name = request.form.get('username')
    email = request.form.get('user_email')
    password = request.form.get('user_password')

    cursor.execute("""INSERT INTO `user` (`name`,`email`,`password`) VALUES
    ('{}', '{}', '{}')""".format(name, email, password))
    conn.commit()

    cursor.execute("""SELECT * FROM `user` WHERE `email` LIKE '{}'""".format(email))
    myuser = cursor.fetchall()
    session['name'] = myuser[0][0]
    return redirect('/home')

@app.route('/add_feed', methods = ['POST'])
def add_feed():
    cursor.execute("""SELECT * FROM `user` """)
    user_feed = cursor.fetchall()

    '''if len(user_feed) > 0:
        session['name'] = user_feed[0][0]
        active_user = session['name']'''
    active_user = session['name']
    feed = request.form.get('feedback')

    cursor.execute("""INSERT INTO `feedback` (`feedbacks`,`name`) VALUES
    ( '{}','{}')""".format(feed, active_user))
    conn.commit()
    print(active_user)
    return redirect('/home')

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    active = session['name']

    if request.method == 'POST':
        user_question = request.form['question']

        response = text_process.chatbot_response(user_question)

        cursor.execute("""INSERT INTO `queries` (`name`,`questions`,`answers`) VALUES
            ('{}', '{}','{}')""".format(active, user_question, response))
        conn.commit()
        print(active)

    return jsonify({"response": response })

@app.route('/logout')
def logout():
    session.pop('name')
    return redirect('/')

@app.route('/adminlogout')
def adminlogout():
    session.pop('email')
    return redirect('/admin')

if __name__ == "__main__":
    app.run(debug=True)