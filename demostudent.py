from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import PIL
from tkinter import messagebox
import mysql.connector
import cv2


class student:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1530x1080+0+0")

        # self.root.minsize(800, 400)
        # self.root.maxsize(1920, 1080)
        # self.root.attributes('-fullscreen', True)

        # declaring Vaariables
        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_id = StringVar()
        self.var_name = StringVar()
        self.var_div = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_phone = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()
        self.var_searchtxt=StringVar()
        self.var_search=StringVar()

        bg_img = Image.open(
            r"D:\Vishwakarma University\SY\ISF\FaceRecognisation\images\student201.jpg")
        bg_img = bg_img.resize((1830, 780), PIL.Image.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(bg_img)

        fir_label = Label(self.root, image=self.photoimage)
        fir_label.place(x=0, y=0, width=1830, height=880)

        title_label = Label(self.root, text="PERSON DETAILS", font=(
            "Arial", 35, "bold"), bg="black", fg="white")
        title_label.place(x=0, y=0, width=1600, height=50)

        main_frame = Frame(fir_label, bd=5, bg="black")
        main_frame.place(x=0, y=0, width=1730, height=780)

        # left label frame
        Left_frame = LabelFrame(main_frame, bd=5, bg="white", relief=RIDGE, text="Person Details", font=(
            "Arial", 15, "bold"))
        Left_frame.place(x=5, y=50, width=750, height=780)

        left_img = Image.open(
            r"d:\Vishwakarma University\SY\ISF\FaceRecognisation\images\left_frame.png")
        left_img = left_img.resize((700, 200), PIL.Image.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(left_img)

        fir_label = Label(Left_frame, image=self.photoimage)
        fir_label.place(x=40, y=10, width=650, height=180)

        # course frame
        Left_frame_course = LabelFrame(Left_frame, bd=5, bg="white", relief=RIDGE, text="Course Details", font=(
            "Arial", 15, "bold"))
        Left_frame_course.place(x=5, y=190, width=750, height=120)

        # departmenrt label
        depart = Label(Left_frame_course, text="Department: ", font=(
            "Arial", 12, "bold"), bg="white")
        depart.grid(row=0, column=0)

        dapart_combo = ttk.Combobox(Left_frame_course, textvariable=self.var_dep, font=(
            "Arial", 12), width=17, state="readonly")
        dapart_combo["values"] = (
            "Select Department", "Engineering", "Pharmacy", "Design", "Commerce")
        dapart_combo.current(0)
        dapart_combo.grid(row=0, column=1, padx=5, pady=10, sticky=W)

        # course
        course = Label(Left_frame_course, text="Course:", font=(
            "Arial", 12, "bold"), bg="white")
        course.grid(row=0, column=2, padx=10, pady=5, sticky=W)

        course_combo = ttk.Combobox(Left_frame_course, textvariable=self.var_course, font=(
            "Arial", 12), width=17, state="readonly")
        course_combo["values"] = ("Select Course", "Computer Science",
                                  "B.Com", "B.Pharma", "D.Pharma", "Fashion", "Interior")
        course_combo.current(0)
        course_combo.grid(row=0, column=3, padx=5, pady=5, sticky=W)

        # Year
        year = Label(Left_frame_course, text="Year:", font=(
            "Arial", 12, "bold"), bg="white")
        year.grid(row=2, column=0, padx=10, pady=5, sticky=W)

        year_combo = ttk.Combobox(Left_frame_course, textvariable=self.var_year, font=(
            "Arial", 12), width=17, state="readonly")
        year_combo["values"] = ("Select Year", "2020",
                                "2021", "2022", "2023", "2024")
        year_combo.current(0)
        year_combo.grid(row=2, column=1, padx=5, pady=5, sticky=W)

        # semester
        semester = Label(Left_frame_course, text="Semester:", font=(
            "Arial", 12, "bold"), bg="white")
        semester.grid(row=2, column=2, padx=10, pady=5, sticky=W)

        semester_combo = ttk.Combobox(Left_frame_course, textvariable=self.var_semester, font=(
            "Arial", 12), width=17, state="readonly")
        semester_combo["values"] = (
            "Select Semester", "Sem 1", "Sem 2", "Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7", "Sem 8")
        semester_combo.current(0)
        semester_combo.grid(row=2, column=3, padx=5, pady=5, sticky=W)

        # course frame student
        Left_frame_student = LabelFrame(Left_frame, bd=5, bg="white", relief=RIDGE, text="Student Information", font=(
            "Arial", 15, "bold"))
        Left_frame_student.place(x=5, y=330, width=750, height=370)

        # Student ID
        std_label = Label(Left_frame_student, text="Student ID:", font=(
            "Arial", 12, "bold"), bg="white")
        std_label.grid(row=0, column=0, padx=10, pady=5, sticky=W)
        std_label_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_id, width=20, font=("Arial", 12))
        std_label_entry.grid(row=0, column=1, padx=10, pady=5, sticky=W)

        # Student Name
        std_name = Label(Left_frame_student, text="Student Name:", font=(
            "Arial", 12, "bold"), bg="white")
        std_name.grid(row=0, column=3, padx=10, pady=5, sticky=W)
        std_name_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_name, width=20, font=("Arial", 12))
        std_name_entry.grid(row=0, column=4, padx=10, pady=5, sticky=W)

        # Student DIVISON
        std_division = Label(Left_frame_student, text="Student Division:", font=(
            "Arial", 12, "bold"), bg="white")
        std_division.grid(row=2, column=0, padx=10, pady=5, sticky=W)

        # std_division_entry = ttk.Entry(
        #    Left_frame_student,textvariable=self.var_div, width=20, font=("Arial", 12))
        # std_division_entry.grid(row=2, column=1, padx=10, pady=5, sticky=W)
        std_division_entry = ttk.Combobox(Left_frame_student, textvariable=self.var_div, font=(
            "Arial", 12), width=20, state="readonly")
        std_division_entry["values"] = (
            "Select option", "A", "B", "C", "D", "E", "F", "G")
        std_division_entry.current(0)
        std_division_entry.grid(row=2, column=1, padx=10, pady=5, sticky=W)

        # Student Roll_No
        std_roll_no = Label(Left_frame_student, text="Student Roll No:", font=(
            "Arial", 12, "bold"), bg="white")
        std_roll_no.grid(row=2, column=3, padx=10, pady=5, sticky=W)
        std_roll_no_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_roll, width=20, font=("Arial", 12))
        std_roll_no_entry.grid(row=2, column=4, padx=10, pady=5, sticky=W)

        # Student Gender
        std_gender = Label(Left_frame_student, text="Gender:", font=(
            "Arial", 12, "bold"), bg="white")
        std_gender.grid(row=3, column=0, padx=10, pady=5, sticky=W)
        # std_gender_entry = ttk.Entry(
        #    Left_frame_student,textvariable=self.var_gender, width=20, font=("Arial", 12))
        # std_gender_entry.grid(row=3, column=1, padx=10, pady=5, sticky=W)
        std_gender_entry = ttk.Combobox(Left_frame_student, textvariable=self.var_gender, font=(
            "Arial", 12), width=20, state="readonly")
        std_gender_entry["values"] = (
            "Select option", "Male", "Female", "other")
        std_gender_entry.current(0)
        std_gender_entry.grid(row=3, column=1, padx=10, pady=5, sticky=W)

        # Student DOB
        std_label = Label(Left_frame_student, text="Date Of Birth:", font=(
            "Arial", 12, "bold"), bg="white")
        std_label.grid(row=3, column=3, padx=10, pady=5, sticky=W)
        std_label_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_dob, width=20, font=("Arial", 12))
        std_label_entry.grid(row=3, column=4, padx=10, pady=5, sticky=W)

        # Student email
        std_email = Label(Left_frame_student, text="Student Email:", font=(
            "Arial", 12, "bold"), bg="white")
        std_email.grid(row=4, column=0, padx=10, pady=5, sticky=W)
        std_email_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_email, width=20, font=("Arial", 12))
        std_email_entry.grid(row=4, column=1, padx=10, pady=5, sticky=W)

        # Student Phone number
        std_contact = Label(Left_frame_student, text="Contact NO:", font=(
            "Arial", 12, "bold"), bg="white")
        std_contact.grid(row=4, column=3, padx=10, pady=5, sticky=W)
        std_contact_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_phone, width=20, font=("Arial", 12))
        std_contact_entry.grid(row=4, column=4, padx=10, pady=5, sticky=W)

        # student address
        std_address = Label(Left_frame_student, text="Address:", font=(
            "Arial", 12, "bold"), bg="white")
        std_address.grid(row=5, column=0, padx=10, pady=5, sticky=W)
        std_address_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_address, width=20, font=("Arial", 12))
        std_address_entry.grid(row=5, column=1, padx=10, pady=5, sticky=W)

        # class teacher
        std_teacher = Label(Left_frame_student, text="Class Teacher:", font=(
            "Arial", 12, "bold"), bg="white")
        std_teacher.grid(row=5, column=3, padx=10, pady=5, sticky=W)
        std_teacher_entry = ttk.Entry(
            Left_frame_student, textvariable=self.var_teacher, width=20, font=("Arial", 12))
        std_teacher_entry.grid(row=5, column=4, padx=10, pady=5, sticky=W)

        # radio button
        self.var_radio1 = StringVar()
        radbtn1 = ttk.Radiobutton(
            Left_frame_student, variable=self.var_radio1, text="Take Photo Sample", value="Yes")
        radbtn1.grid(row=6, column=0)

        radbtn2 = ttk.Radiobutton(
            Left_frame_student, variable=self.var_radio1, text="No Photo Sample", value="No")
        radbtn2.grid(row=6, column=1)

        # Button frame
        btn_frame = LabelFrame(Left_frame_student, bd=5,
                               bg="white", relief=RIDGE)
        btn_frame.place(x=5, y=210, width=750, height=120)

        save_btn = Button(btn_frame, text="Save", command=self.add_data, font=(
            "Arial", 12, "bold"), width=15)
        save_btn.grid(row=0, column=0, padx=10, pady=10)

        update_btn = Button(btn_frame, text="Update", command=self.update_data,
                            font=("Arial", 12, "bold"), width=15)
        update_btn.grid(row=0, column=1, padx=10, pady=10)

        delete_btn = Button(btn_frame, text="Delete", command=self.delete_data,
                            font=("Arial", 12, "bold"), width=15)
        delete_btn.grid(row=0, column=2, padx=10, pady=10)

        reset_btn = Button(btn_frame, text="Reset", command=self.reset_window,
                           font=("Arial", 12, "bold"), width=15)
        reset_btn.grid(row=0, column=3, padx=10, pady=10)

        # btn frame 2
        btn_frame2 = LabelFrame(btn_frame, bd=5, bg="white", relief=RIDGE)
        btn_frame2.place(x=5, y=50, width=750, height=55)

        take_photo_btn = Button(btn_frame2, command=self.generate_dataset, text="Take Photo", font=(
            "Arial", 12, "bold"), width=15)
        take_photo_btn.grid(row=0, column=0, padx=100, pady=0)

        update_photo_btn = Button(
            btn_frame2, text="Update Photo", font=("Arial", 12, "bold"), width=15)
        update_photo_btn.grid(row=0, column=1, padx=10, pady=10)

        # right label frame
        Right_frame = LabelFrame(main_frame, bd=5, bg="white", relief=RIDGE, text="Entered Information", font=(
            "Arial", 15, "bold"))
        Right_frame.place(x=760, y=50, width=758, height=780)

        # image
        right_img = Image.open("images/student202.jpg")
        right_img = right_img.resize((690, 180), Image.LANCZOS)
        self.photoimg_right = ImageTk.PhotoImage(right_img)

        right_label = Label(Right_frame, image=self.photoimg_right)
        right_label.place(x=5, y=5, width=690, height=180)

        # right label search
        search_frame = LabelFrame(Right_frame, bd=5, bg="white", relief=RIDGE, text="Search Details", font=(
            "Arial", 15, "bold"))
        search_frame.place(x=5, y=185, width=750, height=80)

        search = Label(search_frame, text="Search by:", font=(
            "Arial", 12, "bold"), bg="white")
        search.grid(row=0, column=0, padx=10, pady=5, sticky=W)

        search_combo = ttk.Combobox(search_frame, font=(
            "Arial", 12), width=17, state="readonly")
        search_combo["values"] = (
            "Select", "Roll No", "Phone Number")
        search_combo.current(0)
        search_combo.grid(row=0, column=2, padx=5, pady=5, sticky=W)

        search_entry = ttk.Entry(
            search_frame, width=20,textvariable=self.var_searchtxt, font=("Fira Sans", 12,))
        search_entry.grid(row=0, column=3, padx=10, pady=5, sticky=W)

        # buttons

        search_btn_2 = Button(
            search_frame, text="Search",command=self.search_data, font=("Fira Sans", 12, "bold"), width=10, bg="grey")
        search_btn_2.grid(row=0, column=4, padx=10, pady=8)

        show_btn_3 = Button(
            search_frame, text="Show All",command=self.show_all, font=("Fira Sans", 12, "bold"), width=10, bg="grey")
        show_btn_3.grid(row=0, column=5, padx=10, pady=8)

        # table frame
        table_frame = Frame(Right_frame, bd=5, bg="white", relief=RIDGE)
        table_frame.place(x=5, y=270, width=750, height=430)

        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)

        self.student_table = ttk.Treeview(table_frame, columns=("dept", "cou", "year", "sem", "id", "name", "div", "roll",
                                                                "gen", "dob", "email", "phone", "add", "teacher", "photo"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM, fill=tk.X)
        scroll_y.pack(side=RIGHT, fill=tk.Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dept", text="Department")
        self.student_table.heading("cou", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="ID")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("roll", text="Roll NO")
        self.student_table.heading("gen", text="Gender")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("add", text="Address")
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="Photo")
        self.student_table["show"] = "headings"

        self.student_table.column("dept", width=100)
        self.student_table.column("cou", width=100)
        self.student_table.column("year", width=50)
        self.student_table.column("sem", width=50)
        self.student_table.column("id", width=50)
        self.student_table.column("name", width=100)
        self.student_table.column("div", width=50)
        self.student_table.column("roll", width=50)
        self.student_table.column("gen", width=50)
        self.student_table.column("dob", width=100)
        self.student_table.column("email", width=100)
        self.student_table.column("phone", width=100)
        self.student_table.column("add", width=100)
        self.student_table.column("teacher", width=100)
        self.student_table.column("photo", width=100)

        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>", self.get_cursor)
        self.fetch_data()

    # functions

    def add_data(self):
        if self.var_dep.get() == "Select Department" or self.var_name.get() == " " or self.var_id.get() == " ":
            messagebox.showerror(
                "Error", "All Fields are required: ", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="admin",
                    database="face_recognition",
                )
                my_cursor = conn.cursor()
                my_cursor.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                                                                                                        self.var_dep.get(),
                                                                                                        self.var_course.get(),
                                                                                                        self.var_year.get(),
                                                                                                        self.var_semester.get(),
                                                                                                        self.var_id.get(),
                                                                                                        self.var_name.get(),
                                                                                                        self.var_div.get(),
                                                                                                        self.var_roll.get(),
                                                                                                        self.var_gender.get(),
                                                                                                        self.var_dob.get(),
                                                                                                        self.var_email.get(),
                                                                                                        self.var_phone.get(),
                                                                                                        self.var_address.get(),
                                                                                                        self.var_teacher.get(),
                                                                                                        self.var_radio1.get()
                                                                                                        ))
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo(
                    "Success", "Details Added successfully", parent=self.root)
            except Exception as es:
                messagebox.showerror(
                    "Error", f"Due to: {str(es)}", parent=self.root)

    # fetch database

    def fetch_data(self):
        conn = mysql.connector.connect(
            host="localhost",
            username="root",
            password="admin",
            database="face_recognition",
        )
        my_cursor = conn.cursor()
        my_cursor.execute("select * from student")
        data = my_cursor.fetchall()

        if len(data) != 0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("", END, values=i)
            conn.commit()
        conn.close()

    # get DATA on click
    def get_cursor(self, event=""):
        cursor_focus = self.student_table.focus()
        content = self.student_table.item(cursor_focus)
        data = content["values"]

        self.var_dep.set(data[0]),
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data[3]),
        self.var_id.set(data[4]),
        self.var_name.set(data[5]),
        self.var_div.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dob.set(data[9]),
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14]),

    # update DAta

    def update_data(self):
        if self.var_dep.get() == "Select Department" or self.var_name.get() == "" or self.var_id.get() == "":
            messagebox.showerror(
                "Error", "All Fields are required: ", parent=self.root)
        else:
            try:
                update = messagebox.askyesno(
                    "Update", "Do you want to update this student details?", parent=self.root)
                if update > 0:
                    conn = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="admin",
                        database="face_recognition",
                    )
                    my_cursor = conn.cursor()
                    my_cursor.execute("UPDATE student SET Department=%s, Course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll_NO=%s, Gender=%s, DOB=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, Photo=%s WHERE ID=%s", (
                                                                                                                                                                                                                        self.var_dep.get(),
                                                                                                                                                                                                                        self.var_course.get(),
                                                                                                                                                                                                                        self.var_year.get(),
                                                                                                                                                                                                                        self.var_semester.get(),
                                                                                                                                                                                                                        self.var_name.get(),
                                                                                                                                                                                                                        self.var_div.get(),
                                                                                                                                                                                                                        self.var_roll.get(),
                                                                                                                                                                                                                        self.var_gender.get(),
                                                                                                                                                                                                                        self.var_dob.get(),
                                                                                                                                                                                                                        self.var_email.get(),
                                                                                                                                                                                                                        self.var_phone.get(),
                                                                                                                                                                                                                        self.var_address.get(),
                                                                                                                                                                                                                        self.var_teacher.get(),
                                                                                                                                                                                                                        self.var_radio1.get(),  # Fixed this line
                                                                                                                                                                                                                        self.var_id.get()
                                                                                                                                                                                                                                            ))
                else:
                    if not update:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success", "Student data updated successfully updated", parent=self.root)
                
            except Exception as es:
                messagebox.showerror(
                    "Error", f"Due to: {str(es)}", parent=self.root)

    # Delete Data

    def delete_data(self):
        if self.var_id.get == "":
            messagebox.showerror(
                "Error", "Student ID not found", parent=self.root)
        else:
            try:
                delete = messagebox.askyesno(
                    "Delete Student Data", "Do you want to Delete this student details?", parent=self.root)
                if delete > 0:
                    conn = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="admin",
                        database="face_recognition",
                    )
                    my_cursor = conn.cursor()
                    sql = "delete from student where ID=%s"
                    val = (self.var_id.get(),)
                    my_cursor.execute(sql, val)

                else:
                    if not delete:
                        return

                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo(
                    "Success", "Student Details Deleated Successfully", parent=self.root)
            except Exception as es:
                messagebox.showerror(
                    "Error", f"Due to: {str(es)}", parent=self.root)

    # reset Window

    def reset_window(self):
        self.var_dep.set("Select Department")
        self.var_course.set("Select Course")
        self.var_year.set("Select Year")
        self.var_semester.set("Select Semester")
        self.var_id.set("")
        self.var_name.set("")
        self.var_div.set("Select option")
        self.var_roll.set("")
        self.var_gender.set("Select option")
        self.var_dob.set("")
        self.var_email.set("")
        self.var_phone.set("")
        self.var_address.set("")
        self.var_teacher.set("")
        self.var_radio1.set("")

    # take photo sample

    def generate_dataset(self):
        if self.var_dep.get()=="Select Department" or self.var_name.get()=="" or self.var_id.get()=="":
                messagebox.showerror("Error","All fields are required",parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="admin",
                    database="face_recognition")
                
                my_cursor=conn.cursor()
                my_cursor.execute("Select * from student")
                myresult=my_cursor.fetchall()
                id=0
                for x in myresult:
                    id+=1
                my_cursor.execute("Update student set Department=%s, Course=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll_NO=%s, Gender=%s, DOB=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, Photo=%s WHERE ID=%s",(
                                                                                                                                                                                    self.var_dep.get(),
                                                                                                                                                                                                                        self.var_course.get(),
                                                                                                                                                                                                                        self.var_year.get(),
                                                                                                                                                                                                                        self.var_semester.get(),
                                                                                                                                                                                                                        self.var_name.get(),
                                                                                                                                                                                                                        self.var_div.get(),
                                                                                                                                                                                                                        self.var_roll.get(),
                                                                                                                                                                                                                        self.var_gender.get(),
                                                                                                                                                                                                                        self.var_dob.get(),
                                                                                                                                                                                                                        self.var_email.get(),
                                                                                                                                                                                                                        self.var_phone.get(),
                                                                                                                                                                                                                        self.var_address.get(),
                                                                                                                                                                                                                        self.var_teacher.get(),
                                                                                                                                                                                                                        self.var_radio1.get(),  # Fixed this line
                                                                                                                                                                                                                        self.var_id.get()==id+1
                                                                                                                                                                                                        ))
                conn.commit()
                self.fetch_data()
                conn.close()
                
                #========Load predefined data on frontal face from open cv=========
                face_classifiers=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

                def face_cropped(img):
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces=face_classifiers.detectMultiScale(gray,1.3,5)
                    #scaling factor=1.3
                    #minimum neighbour=5

                    for(x,y,w,h) in faces:
                        face_cropped=img[y:y+h,x:x+w]
                        return face_cropped

                cap=cv2.VideoCapture(0)
                #Videocapture(0) for inbuilt camera
                #Videocapture(1) for external webcam
                img_id=0
                while True:
                    ret,my_frame=cap.read()
                    if face_cropped(my_frame) is not None:
                        img_id+=1
                        face=cv2.resize(face_cropped(my_frame),(450,450))
                        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        file_name_path="data/user."+str(id)+"."+str(img_id)+".jpg"
                        cv2.imwrite(file_name_path,face)
                        cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                        cv2.imshow("Cropped Face",face)

                    if cv2.waitKey(1)==13 or int(img_id)==100:
                        break

                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Result","Generating Data sets Completed!!",parent=self.root)
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)







    def search_data(self):
        if self.var_searchtxt.get() == "" or self.var_search.get() == "Select Option":
            messagebox.showerror("Error", "Select Combo option and enter entry box", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(
                    host="localhost",
                    username="root",
                    password="admin",
                    database="face_recognition"
                )
                my_cursor = conn.cursor()

                # Use parameterized query to avoid SQL injection
                search_field = self.var_search.get()
                search_text = '%' + self.var_searchtxt.get() + '%'
                query = f"SELECT * FROM student1 WHERE {search_field} LIKE %s"
            
                my_cursor.execute(query, (search_text,))
                rows = my_cursor.fetchall()


                if len(data)!=0:
                        self.student_table.delete(*self.student_table.get_children())
                        for i in data:
                            self.student_table.insert("",END,values=i)
                        conn.commit()
                        conn.close()
                else:
                            messagebox.showerror("Error", "Data Not Found", parent=self.root)

                            conn.close()

            except Exception as es:
                    messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)



            # show all 
    def show_all(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="admin",database="face_recognition")
        # conn=mysql.connector.connect(host="localhost",username="root",password="Keshav@123",database="mydata")
        my_cursor=conn.cursor()
        my_cursor.execute("select * from student1")
        data=my_cursor.fetchall()

        if len(data)!=0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("",END,values=i)
            conn.commit()
        conn.close()




if __name__ == "__main__":
    root = Tk()
    obj = student(root)
    root.mainloop()
