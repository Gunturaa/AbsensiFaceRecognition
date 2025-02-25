from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash

import mysql.connector
import cv2
from PIL import Image
import numpy as np

import os
import time
from datetime import date
import pandas as pd
from io import BytesIO


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Change this to a secure random key in production


cnt = 0
pause_cnt = 0
justscanned = False

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "D:/projekku/AbsensiFaceRecognition/resources/haarcascade_frontalface_default.xml")

    if face_classifier.empty():
        raise IOError("Error loading face cascade classifier. Check the path to the XML file.")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "D:/projekku/AbsensiFaceRecognition/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('/')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        global justscanned
        global pause_cnt

        pause_cnt += 1

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (153, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)

                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('" + str(
                        date.today()) + "', '" + pnbr + "')")
                    mydb.commit()

                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "D:/projekku/AbsensiFaceRecognition/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 400, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user exists
        mycursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = mycursor.fetchone()
        
        if user and check_password_hash(user[2], password):
            session['username'] = username
            session['role'] = user[3]
            
            # Redirect based on role
            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('employee_dashboard'))



        
        flash('Invalid username or password', 'error')
        return redirect(url_for('login'))
        
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        # Check if user already exists
        mycursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = mycursor.fetchone()
        
        if user:
            flash('Username already exists!', 'error')
            return redirect(url_for('register'))
            
        try:
            # Hash password and insert new user
            hashed_password = generate_password_hash(password)

            mycursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", 
                          (username, hashed_password, role))
        except Exception as e:
            flash('Error during registration. Please try again.', 'error')
            app.logger.error(f"Registration error: {str(e)}")
            return redirect(url_for('register'))

        mydb.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

        
    return render_template('register.html')


@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    try:
        # Debug: Check database connection
        if not mydb.is_connected():
            mydb.reconnect()
            app.logger.info("Reconnected to database")
        
        # Debug: Check users table exists
        mycursor.execute("SHOW TABLES LIKE 'users'")
        if not mycursor.fetchone():
            flash('Users table does not exist', 'error')
            return redirect(url_for('admin_dashboard'))
        
        # Get all users from database with proper column selection
        mycursor.execute("SELECT id, username, role FROM users")
        users = mycursor.fetchall()
        
        # Get all personnel data
        mycursor.execute("SELECT prs_nbr, prs_name, prs_skill, prs_active, prs_added FROM prs_mstr")
        personnel = mycursor.fetchall()
        
        if not users:
            flash('No users found in database', 'warning')
        else:
            app.logger.info(f"Found {len(users)} users in database")
        
        if not personnel:
            flash('No personnel data found', 'warning')
        else:
            app.logger.info(f"Found {len(personnel)} personnel records")
        
        return render_template('admin_dashboard.html', users=users, personnel=personnel)

    except Exception as e:
        app.logger.error(f"Error fetching users: {str(e)}")
        flash('Error loading user data', 'error')
        return redirect(url_for('admin_dashboard'))



@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form['username']
        role = request.form['role']
        
        mycursor.execute("UPDATE users SET username = %s, role = %s WHERE id = %s",
                        (username, role, user_id))
        mydb.commit()
        flash('User updated successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    
    mycursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = mycursor.fetchone()
    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
        
    mycursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    mydb.commit()
    flash('User deleted successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/edit_personnel/<int:personnel_id>', methods=['GET', 'POST'])
def edit_personnel(personnel_id):
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form['name']
        skill = request.form['skill']
        active = 1 if request.form.get('active') else 0
        
        mycursor.execute("UPDATE prs_mstr SET prs_name = %s, prs_skill = %s, prs_active = %s WHERE prs_nbr = %s",
                        (name, skill, active, personnel_id))
        mydb.commit()
        flash('Personnel updated successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    
    mycursor.execute("SELECT * FROM prs_mstr WHERE prs_nbr = %s", (personnel_id,))
    personnel = mycursor.fetchone()
    return render_template('edit_personnel.html', personnel=personnel)

@app.route('/delete_personnel/<int:personnel_id>')
def delete_personnel(personnel_id):
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
        
    mycursor.execute("DELETE FROM prs_mstr WHERE prs_nbr = %s", (personnel_id,))
    mydb.commit()
    flash('Personnel deleted successfully', 'success')
    return redirect(url_for('admin_dashboard'))



@app.route('/employee')
def employee_dashboard():
    if 'username' not in session or session['role'] != 'employee':
        return redirect(url_for('login'))
    
    try:
        # Get personnel data
        mycursor.execute("SELECT prs_nbr, prs_name, prs_skill, prs_active, prs_added FROM prs_mstr")
        personnel = mycursor.fetchall()
        
        if not personnel:
            flash('No personnel data found', 'warning')
        else:
            app.logger.info(f"Found {len(personnel)} personnel records")
        
        return render_template('employee_dashboard.html', personnel=personnel)
    except Exception as e:
        app.logger.error(f"Error fetching personnel data: {str(e)}")
        flash('Error loading personnel data', 'error')
        return redirect(url_for('employee_dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/')
def home():



    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)


@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('fr_page.html', data=data)


@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})


@app.route('/loadData', methods=['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return jsonify(response=data)

@app.route('/download_today_scan')
def download_today_scan():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    
    try:
        # Ensure database connection is active
        if not mydb.is_connected():
            mydb.reconnect()
        
        # Get today's scan data
        mycursor.execute("""
            SELECT a.accs_id, a.accs_prsn, a.accs_date, 
                   DATE_FORMAT(a.accs_added, '%Y-%m-%d %H:%i:%s') as accs_added,
                   b.prs_name, b.prs_skill
            FROM accs_hist a
            LEFT JOIN prs_mstr b ON a.accs_prsn = b.prs_nbr
            WHERE a.accs_date = CURDATE()
            ORDER BY a.accs_id DESC
        """)
        data = mycursor.fetchall()
        
        if not data:
            flash('No scan data found for today', 'warning')
            return redirect(url_for('admin_dashboard'))
        
        # Create DataFrame with all accs_hist columns
        df = pd.DataFrame(data, columns=[
            'ID', 
            'Personnel ID', 
            'Date', 
            'Scan Time', 
            'Name', 
            'Skill'
        ])

        
        # Create Excel file in memory with formatting
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Today Scan')
            
            # Get workbook and worksheet references
            workbook = writer.book
            worksheet = writer.sheets['Today Scan']
            
            # Add header formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 10)  # ID
            worksheet.set_column('B:B', 15)  # Personnel ID
            worksheet.set_column('C:C', 12)  # Date
            worksheet.set_column('D:D', 15)  # Scan Time
            worksheet.set_column('E:E', 20)  # Name
            worksheet.set_column('F:F', 20)  # Skill
            
            # Add autofilter
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)




        
        output.seek(0)
        
        # Create response with Excel file
        today = date.today().strftime('%Y-%m-%d')
        filename = f"today_scan_{today}.xlsx"
        
        response = Response(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
        

        
    except Exception as e:
        app.logger.error(f"Error generating Excel file: {str(e)}")
        flash('Error generating download file', 'error')
        return redirect(url_for('admin_dashboard'))



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
