import sys
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from untitled import Ui_Form
import 登录页面
import 注册
import 用户管理
from predict import predict_single

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PredictThread(QThread):
    """图像预测线程"""
    finished = pyqtSignal(str, bool)  # 输出路径, 是否成功
    progress = pyqtSignal(str)  # 进度消息
    
    def __init__(self, filepath, save_dir):
        super().__init__()
        self.filepath = filepath
        self.save_dir = save_dir
    
    def run(self):
        try:
            self.progress.emit("分割中...")
            
            # 使用 predict_single 进行预测
            imgs = predict_single(self.filepath)
            
            # 保存结果
            output_filename = os.path.basename(self.filepath)
            output_path = os.path.join(self.save_dir, output_filename)
            imgs.save(output_path)
            
            if os.path.exists(output_path):
                self.progress.emit("分割完成 保存到数据库中")
                self.finished.emit(output_path, True)
            else:
                self.progress.emit("执行失败 请重试")
                self.finished.emit("", False)
        except Exception as e:
            self.progress.emit(f"执行失败: {str(e)}")
            self.finished.emit("", False)


class mywindow(QWidget,登录页面.Ui_Form):    #登录页面
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        path = str(os.getcwd()) + "\\users.json"
        if os.path.exists(path):    #设置颜色
            with open('users.json', mode='r', encoding='utf-8')as f:
                text = f.read()
                self.users = json.loads(text)
                f.close()
        else:
            a = [{"name": "admin123", "username": "admin123", "password": "admin123", "values": "\u7ba1\u7406\u5458"}]
            json_str = json.dumps(a)  # 写入json文件
            with open('users.json', 'w', encoding='utf-8') as f:
                f.write(json_str)
                f.close()
            self.users = a
        self.quitbt.clicked.connect(self.close)
        self.namelabel.setStyleSheet("color:yellow")    #设置颜色
        self.usersedit.setPlaceholderText("请输入账号")
        self.pwdetid.setPlaceholderText("请输入密码")
        self.pwdetid.setEchoMode(QLineEdit.Password)
        self.pwdetid.returnPressed.connect(self.loginit)
        self.login.setFixedHeight(60)
        self.setFixedSize(self.width(),self.height())
        self.count = 0

    def changeuser(self):
        self.close()
        mychange.show()
        mychange.settitle(self.usersedit.text(),self.pwdetid.text())

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./images/图像2.jpg")  ## ""中输入图片路径
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    def get_user(self,user):
        with open('users.json', mode='r', encoding='utf-8') as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if user == i["username"]:
                return i
        return False

    def loginit(self):
        user = self.usersedit.text()
        pwd = self.pwdetid.text()
        if user and pwd and self.count < 3:
            users = self.get_user(user)
            if users:
                if pwd == users['password']:
                    self.count = 0
                    myshow.close()
                    mywin.settitle(user)
                    mywin.show()
                else:
                    QMessageBox.information(self, '提示', '密码错误')
                    self.count += 1
                    if self.count >= 3:
                        QMessageBox.information(self, '提示', '密码错误3次，将退出')
                        self.close()
            else:
                QMessageBox.information(self, '提示', '账号不存在')
        else:
            QMessageBox.information(self, '提示', '输入账号密码')

    def register(self):
        self.close()
        myregis.show()

class winregister(QWidget,注册.Ui_Form): #注册页面
    def __init__(self):
        super(winregister,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.quitit.clicked.connect(self.quit)
        self.addnew.clicked.connect(self.ister)
        self.setFixedSize(self.width(),self.height())

    def quit(self): #退出
        self.close()
        myshow.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./images/regster.jpeg")  ## ""中输入图片路径
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    def check_alpha(self,input):    #判断密码是否是数字或者字母
        alphas = 0
        alpha_list = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z" \
                     " a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9 - _".split()
        for i in input:
            alphas = 0
            if i in alpha_list:
                alphas += 1
        if alphas > 0:
            return True
        else:
            return False

    def checkuser(self,user):
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if i["username"] == user:
                return False
        return True

    def getid(self):
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        alist = []
        for i in self.users:
            if "学号" in i.keys():
                alist.append(i["学号"])
        if alist:
            alist.sort()
            num = alist[-1]
            num += 1
            while True:
                if num not in alist:
                    return num
                else:
                    num += 1
        else:
            return 100001

    def ister(self):
        name = self.name.text()
        username = self.username.text()
        pwd = self.password.text()
        pwd1 = self.okpassword.text()
        value = self.comboBox.currentText()
        if value == "学生":
            num = self.getid()
            self.a = {"name": name, "username": username, "password": pwd,"values":value,"选课":[],"成绩":[],"学号":num}
        elif value == "管理员":
            self.a = {"name": name, "username": username, "password": pwd, "values": value}
        elif value == "老师":
            self.a = {"name": name, "username": username, "password": pwd, "values": value,"科目":""}
        res = False
        if pwd == pwd1:
            temp = self.checkuser(username)
            if temp:
                if len(username) >= 8 and len(username) <= 18:
                    if len(pwd) >= 8 and len(pwd) <= 18:
                        res = self.check_alpha(pwd)
                    else:
                        QMessageBox.information(self,'警告', "密码长度不足8位或者超过18位")
                else:
                    QMessageBox.information(self,'警告', "账号长度不足8位或者超过18位")
            else:
                QMessageBox.information(self,'警告', "账号已存在")
        else:
            QMessageBox.information(self, '警告', "两次密码不相同")
        if res:
            self.add_person(self.a)

    def add_person(self, adic): #注册账号写入信息
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        self.users.append(adic)  # 获取里面的内容
        json_str = json.dumps(self.users)  # 写入json文件
        with open('users.json', 'w',encoding='utf-8') as f:
            f.write(json_str)
            f.close()
        QMessageBox.information(self, '提示', "注册成功，点击确定后登录")
        self.close()
        myshow.show()

class changeuser(QWidget,用户管理.Ui_Form):
    def __init__(self):
        super(changeuser,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setFixedSize(self.width(),self.height())
        self.yanzhengpwd.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.password.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.username.setPlaceholderText("请输入用户名")
        self.password.setPlaceholderText("请输入原始密码")
        self.newpassword.setPlaceholderText("请输入新密码")
        self.newpassword2.setPlaceholderText("请再次输入新密码")
        self.yanzhenguser.setPlaceholderText("如未输入账号密码")
        self.yanzhengpwd.setPlaceholderText("此处输入账号密码验证")
        self.changeokbt.setFixedHeight(40)
        self.deletebt.setFixedHeight(40)
        self.changeokbt.clicked.connect(self.changeit)
        self.deletebt.clicked.connect(self.deleteuser)
        self.comboBox.currentIndexChanged.connect(self.showinformation)
        self.yanzhengpwd.editingFinished.connect(self.yanzheng)
        self.pushButton.clicked.connect(self.returnlogin)
        self.temp = False

    def returnlogin(self):
        self.close()
        myshow.show()

    def yanzheng(self):
        if self.temp:
            QMessageBox.information(self, "提示", "验证成功")
            self.yanzhenguser.setPlaceholderText("验证成功")
            self.yanzhengpwd.setPlaceholderText("可直接删除用户")
            self.yanzhengpwd.setText("")
            self.yanzhenguser.setText("")
        else:
            user = self.yanzhenguser.text()
            pwd = self.yanzhengpwd.text()
            temp = self.checklogin(user,pwd)
            if temp:
                self.temp = True
                QMessageBox.information(self,"提示","验证成功")
                self.yanzhenguser.setPlaceholderText("验证成功")
                self.yanzhengpwd.setPlaceholderText("可直接删除用户")
                alist = self.returnall()
                self.comboBox.clear()
                self.comboBox.addItems(alist)
                self.yanzhengpwd.setText("")
                self.yanzhenguser.setText("")

            else:
                QMessageBox.information(self, "提示", "验证失败！！！\n"
                                                    "账号或密码不正确")
                self.yanzhenguser.setPlaceholderText("如未输入账号密码")
                self.yanzhengpwd.setPlaceholderText("此处输入账号密码验证")

    def showinformation(self):
        name = self.comboBox.currentText()
        adic = self.returnone(name)
        QMessageBox.information(self, "个人信息","用户名：%s\n账号：%s\n类型：%s"%(adic["name"],adic["username"],adic["values"]))

    def check_alpha(self,input):    #判断密码是否是数字或者字母
        alphas = 0
        alpha_list = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z" \
                     " a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9 - _".split()
        for i in input:
            alphas = 0
            if i in alpha_list:
                alphas += 1
        if alphas > 0:
            return True
        else:
            return False

    def change_person(self, name, pwd): #注册账号写入信息
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if i["username"] == name:
                a = i
                a["password"] = pwd
                self.users.remove(i)
                self.users.append(a)
                json_str = json.dumps(self.users)  # 写入json文件
                with open('users.json', 'w',encoding='utf-8') as f:
                    f.write(json_str)
                    f.close()
                self.newpassword2.setText("")
                self.newpassword.setText("")
                self.password.setText("")
                self.username.setText("")
                QMessageBox.information(self, '提示', "修改密码成功")
                break

    def changeit(self):
        user = self.username.text()
        pwd = self.password.text()
        if user and pwd:
            temp = self.checklogin(user,pwd)
            if temp:
                newpwd = self.newpassword.text()
                newpwd1 = self.newpassword2.text()
                if newpwd == newpwd1:
                    if len(pwd) >= 8 and len(pwd) <= 18:
                        self.res = self.check_alpha(pwd)
                        if self.res:
                            self.change_person(user, newpwd)  # 写入json并登录
                        else:
                            QMessageBox.information(self, '警告', "账号密码只能以字母数字_-命名")
                    else:
                        QMessageBox.information(self, '警告', "密码长度不足8位或者超过18位")
                else:
                    QMessageBox.information(self, "错误提示", "两次新密码输入不一致")
            else:
                QMessageBox.information(self,"错误提示","账号或密码错误")
        else:
            QMessageBox.information(self, "错误提示", "先输入账号密码和新密码再修改")

    def returnone(self,user):   #返回个人信息
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if i["username"] == user:
                return i

    def checkvalue(self,user):  #判断管理员权限
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if i["username"] == user:
                if i["values"] == "管理员":
                    return True
                return False
        return False

    def checklogin(self,user,pwd):  #判断登录
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
                if user == i['username']:
                    if pwd == i['password']:
                        return True
                    return False
        return False

    def returnname(self,user):      #返回姓名
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
           if i["username"] == user:
               return i["name"]
        return False

    def deleteuser(self):
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        user = self.comboBox.currentText()
        if user == "admin":
            QMessageBox.information(self, "提示", "管理员账号无法删除")
        else:
            self.comboBox.currentIndexChanged.disconnect()
            for i in self.users:
                if i["username"] == user:
                    self.users.remove(i)
                    json_str = json.dumps(self.users)  # 写入json文件
                    with open('users.json', 'w', encoding='utf-8') as f:
                        f.write(json_str)
                        f.close()
            alist = self.returnall()
            self.comboBox.clear()
            self.comboBox.addItems(alist)
            self.comboBox.currentIndexChanged.connect(self.showinformation)
            QMessageBox.information(self,"提示","删除用户（%s）成功" % user)

    def returnall(self):
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        alist = []
        for i in self.users:
            if i["username"] != "admin":
                alist.append(i["username"])
        return alist

    def settitle(self,user,pwd):
        self.username.setText(user)
        self.password.setText(pwd)
        if user:
            name = self.returnname(user)
            if name:
                self.setWindowTitle("用户管理页面 欢迎你：%s"% name)
                temp = self.checklogin(user,pwd)
                if temp:
                    temp1 = self.checkvalue(user)
                    if temp1:
                        self.yanzhenguser.setText(user)
                        self.yanzhengpwd.setText(pwd)
                        self.yanzheng()

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./images/change.jpg")  ## ""中输入图片路径
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

class mainwin(QWidget,Ui_Form): #主页面
    def __init__(self):
        super(mainwin,self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        
        self.textEdit.setReadOnly(True)
        self.filepath = ""
        self.savefilepath = ""
        self.label.setScaledContents(True)
        self.label_2.setScaledContents(True)
        self.setFixedSize(self.width(),self.height())
        
        # 预测线程
        self.predict_thread = None

        # 这是输入图片
        self.view = QGraphicsView(self)
        self.view.setGeometry(30, 80, 460, 660)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # 这是输入图片滑窗
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setGeometry(100, 710, 300, 20)
        self.slider.setRange(1, 500)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.zoom_image)

        # 这是分割好的图片
        self.view_output = QGraphicsView(self)
        self.view_output.setGeometry(550, 80, 460, 660)
        self.scene_output = QGraphicsScene()
        self.view_output.setScene(self.scene_output)

        # 这是分割好的图片滑窗
        self.slider_output = QSlider(Qt.Horizontal, self)
        self.slider_output.setGeometry(600, 710, 300, 20)
        self.slider_output.setRange(1, 500)
        self.slider_output.setValue(100)
        self.slider_output.valueChanged.connect(self.zoom_output_image)
        
        # 应用现代化样式（在所有控件创建之后）
        self.apply_modern_style()
        
        # 连接信号
        self.pushButton.clicked.connect(self.addjpg)
        self.pushButton_3.clicked.connect(self.rezero)
        self.pushButton_2.clicked.connect(self.start)
        self.pushButton_5.clicked.connect(self.savepath)
        self.pushButton_6.clicked.connect(self.help)
        self.pushButton_7.clicked.connect(self.guanyu)
        self.pushButton_8.clicked.connect(self.close)

    def apply_modern_style(self):
        """应用现代化UI样式"""
        # 主窗口样式
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: "Microsoft YaHei UI", "微软雅黑", sans-serif;
            }
            QGroupBox {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
                font-weight: bold;
                font-size: 12pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #2196F3;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
                font-size: 10pt;
                color: #333333;
            }
            QLabel {
                background-color: #fafafa;
                border: 1px dashed #cccccc;
                border-radius: 6px;
                color: #999999;
                font-size: 11pt;
            }
        """)
        
        # 功能按钮样式（选择、分割、清空）
        button_style = """
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1976D2, stop:1 #1565C0);
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        self.pushButton.setStyleSheet(button_style)
        self.pushButton_2.setStyleSheet(button_style.replace("#2196F3", "#4CAF50").replace("#1976D2", "#388E3C").replace("#42A5F5", "#66BB6A").replace("#1565C0", "#2E7D32"))
        self.pushButton_3.setStyleSheet(button_style.replace("#2196F3", "#FF9800").replace("#1976D2", "#F57C00").replace("#42A5F5", "#FFA726").replace("#1565C0", "#E65100"))
        
        # 顶部菜单按钮样式
        menu_button_style = """
            QPushButton {
                background-color: #ffffff;
                color: #555555;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 6px 15px;
                font-size: 10pt;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #2196F3;
                color: white;
                border: 1px solid #2196F3;
            }
            QPushButton:pressed {
                background-color: #1976D2;
            }
        """
        
        self.pushButton_5.setStyleSheet(menu_button_style)
        self.pushButton_6.setStyleSheet(menu_button_style)
        self.pushButton_7.setStyleSheet(menu_button_style)
        self.pushButton_8.setStyleSheet(menu_button_style.replace("#2196F3", "#F44336").replace("#1976D2", "#D32F2F"))
        
        # 滑块样式
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #e0e0e0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                border: 1px solid #1565C0;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #42A5F5, stop:1 #2196F3);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #64B5F6, stop:1 #2196F3);
                border-radius: 3px;
            }
        """
        
        self.slider.setStyleSheet(slider_style)
        self.slider_output.setStyleSheet(slider_style)
        
        # GraphicsView 样式
        view_style = """
            QGraphicsView {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fafafa;
            }
        """
        self.view.setStyleSheet(view_style)
        self.view_output.setStyleSheet(view_style)

    def zoom_output_image(self, value):
        factor = value / 100.0  # 将滑块值转换为缩放因子
        self.view_output.setTransform(QTransform().scale(factor, factor))


    def zoom_image(self, value):
        factor = value / 100.0  # 将滑块值转换为缩放因子
        self.view.setTransform(QTransform().scale(factor, factor))

    def help(self):
        QMessageBox.information(self,"帮助","配置：配置保存图像的路径\n"
                                            "帮助：查看各功能导航\n"
                                            "关于：查看本系统开发信息\n"
                                            "退出：退出本系统\n"
                                            "选择图像：选择您待分割的图像\n"
                                            "开始分割：选择图像后才可以执行分割操作\n"
                                            "清空：清空页面上已经完成的所有信息\n"
                                            "状态日志：记录你的操作")

    def guanyu(self):
        QMessageBox.information(self,"关于","开发语言：Python\n"
                                            "GUI开发：PyQt5\n"
                                            "项目名称：基于U-Net的眼底血管图像分割系统\n"
                                            )

    def savepath(self):
        path = os.getcwd()
        self.savefilepath = QFileDialog.getExistingDirectory(self, "请选择保存路径", path )
        if self.savefilepath:
            self.textEdit.append("当前保存路径：%s" % self.savefilepath)


    def addjpg(self):
        path = os.getcwd()
        self.imagepath = QFileDialog.getOpenFileName(self, "请选择图像", path,
                                                     "Text Files *.png;*.tif; *.jpg;*.jpeg;;All Files (*)")
        self.filepath = self.imagepath[0]
        # if self.filepath:
        #     self.label.setPixmap(QPixmap(self.filepath))
        #     self.textEdit.append("选择待分割图像：%s"%self.filepath)
        if self.filepath:
            pixmap = QPixmap(self.filepath)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.clear()  # 清除场景中的所有项目
            self.scene.addItem(pixmap_item)

    def on_predict_progress(self, message):
        """更新预测进度"""
        self.textEdit.append(message)
    
    def on_predict_finished(self, output_path, success):
        """预测完成回调"""
        if success and os.path.exists(output_path):
            try:
                # 显示结果
                img = Image.open(output_path)
                imgs1 = np.array(img) * 255
                gray = cv2.cvtColor(imgs1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                cv2.imwrite('temp_output.png', gray)
                
                pixmap_output = QPixmap('temp_output.png')
                pixmap_item_output = QGraphicsPixmapItem(pixmap_output)
                self.scene_output.clear()
                self.scene_output.addItem(pixmap_item_output)
            except Exception as e:
                self.textEdit.append(f"显示结果失败: {str(e)}")
        elif not success:
            QMessageBox.information(self, "提示", "分割失败")
        
        self.pushButton_2.setDisabled(False)

    def start(self):
        """开始分割"""
        try:
            if self.filepath:
                # 确定保存目录
                if self.savefilepath:
                    outputs_dir = self.savefilepath
                else:
                    outputs_dir = os.path.join(os.getcwd(), "out_file")
                    if not os.path.exists(outputs_dir):
                        os.makedirs(outputs_dir)
                
                self.pushButton_2.setDisabled(True)
                
                # 创建并启动预测线程
                self.predict_thread = PredictThread(self.filepath, outputs_dir)
                self.predict_thread.progress.connect(self.on_predict_progress)
                self.predict_thread.finished.connect(self.on_predict_finished)
                self.predict_thread.start()
            else:
                QMessageBox.information(self, "提示", "未加载待分割图像")
        except Exception as e:
            self.textEdit.append(f"启动失败: {str(e)}")
            self.pushButton_2.setDisabled(False)

    def rezero(self):
        self.label.clear()
        self.label_2.clear()
        self.label.setText("未选择")
        self.label_2.setText("未选择")
        self.textEdit.clear()
        self.filepath = ""

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("主页面.jpeg")  ## ""中输入图片路径
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    def settitle(self,user):
        with open('users.json', mode = 'r', encoding='utf-8')as f:
            text = f.read()
            self.users = json.loads(text)
            f.close()
        for i in self.users:
            if i["username"] == user:
                value = i["values"]
                break
        self.user = user
        self.setWindowTitle("分割视网膜血管的眼底病变预测系统 欢迎你：%s"% (user))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.keys()[2])
    
    myshow = mywindow()
    myregis = winregister()
    mywin = mainwin()
    mychange = changeuser()
    
    myshow.login.clicked.connect(myshow.loginit)
    myshow.registerbt.clicked.connect(myshow.register)
    myshow.changebt.clicked.connect(myshow.changeuser)
    
    myshow.show()
    sys.exit(app.exec_())
