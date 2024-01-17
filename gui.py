import customtkinter
import shutil
import os
import subprocess
from invoke import task
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

from src.input_data import InputData

class Register(customtkinter.CTkFrame):
    def __init__(self, master, cmd=None, **kwargs):
        super().__init__(master, **kwargs)

        self.cmd = cmd

        try: 
            _ = customtkinter.CTkFont(family="family name", size=20, weight="bold")
            title_family = "family name"
            subtitle_family = "family name"
        except:
            title_family = None
            subtitle_family = None

        self.subtitle_font = customtkinter.CTkFont(family=subtitle_family, size=20, weight="bold")
        self.register_title = customtkinter.CTkLabel(master=self, text="Đăng ký học sinh mới:", font=self.subtitle_font, fg_color="transparent", text_color="#3E70B5")
        self.register_title.pack(padx=10, pady=10)

        self.text_font = customtkinter.CTkFont(family=title_family, size=20, weight="bold")
        self.username_inp = customtkinter.CTkEntry(master=self, placeholder_text="Nhập ID học sinh", width=250, font=self.text_font)
        self.username_inp.pack(padx=10, pady=10)

        self.register_btn = customtkinter.CTkButton(master=self, text="Đăng ký", font=self.text_font, command=self.add_user)
        self.register_btn.pack(padx=10, pady=10, ipadx=5, ipady=5)

        self.remove_btn = customtkinter.CTkButton(master=self, text="Xóa dữ liệu học sinh", font=self.text_font, command=self.remove_user, hover_color="#737373", fg_color="#808080")
        self.remove_btn.pack(padx=10, pady=10, ipadx=5, ipady=5)


    def add_user(self):
        self.run_model()

        self.message = customtkinter.CTkLabel(master=self, text="Đăng ký học sinh thành công!", font=self.text_font, fg_color="transparent", text_color="#3E70B5")
        self.message.pack(padx=10, pady=10)

        self.unlock_btn = customtkinter.CTkButton(master=self, text="Điểm danh ngay bây giờ", font=self.text_font, command=self.test_model)
        self.unlock_btn.pack(padx=10, pady=10, ipadx=5, ipady=5)

    def run_model(self):
        username = self.username_inp.get()
        new_user = InputData(source=0, 
                            output_path="Dataset/FaceData/raw", 
                            user_id=username, 
                            num_of_img=16, 
                            percent_test=20,
                            augment=True)
        new_user.run()
        os.system('inv train')

    def test_model(self):
        os.system('inv checkface')

    def remove_user(self):
        username = self.username_inp.get().upper()
        try:
            shutil.rmtree(f"Dataset/FaceData/raw/train/{username}")
            shutil.rmtree(f"Dataset/FaceData/raw/test/{username}")
        except Exception:
            pass


class Home(customtkinter.CTkFrame):
    def __init__(self, master, cmd=None, **kwargs):
        super().__init__(master, **kwargs)

        self.cmd = cmd

        try: 
            _ = customtkinter.CTkFont(family="Inconsolata", size=20, weight="bold")
            title_family = "Inconsolata"
            subtitle_family = "Inconsolata ExtraExpanded"
        except:
            title_family = None
            subtitle_family = None

        title_font = customtkinter.CTkFont(family=subtitle_family, size=60, weight="bold", slant="roman")
        self.home_title = customtkinter.CTkLabel(master=self, text="Điểm danh học sinh", font=title_font, fg_color="transparent", text_color="#007ACC")
        self.home_title.pack(padx=10, pady=0)

        self.subtitle_font = customtkinter.CTkFont(family=subtitle_family, size=20)

        self.text_font = customtkinter.CTkFont(family=title_family, size=20, weight="bold")
        self.start_btn = customtkinter.CTkButton(master=self, text="Đăng ký hoặc xóa data học sinh", font=self.text_font, command=self.register_user)
        self.start_btn.pack(padx=10, pady=20, ipadx=10, ipady=10)

        self.or_label = customtkinter.CTkLabel(master=self, text="Hoặc", font=self.subtitle_font, fg_color="transparent", text_color="#3E70B5")
        self.or_label.pack(padx=10, pady=0)

        self.unlock_btn = customtkinter.CTkButton(master=self, text="Điểm danh học sinh", font=self.text_font, command=self.test_model)
        self.unlock_btn.pack(padx=10, pady=20, ipadx=10, ipady=10)

    def register_user(self):
        # self.home_subtitle.pack_forget()
        self.start_btn.pack_forget()
        self.or_label.pack_forget()
        self.unlock_btn.pack_forget()

        self.back_btn = customtkinter.CTkButton(master=self, text="Trở về trang chính", font=self.text_font, command=self.back_home, hover_color="#737373", fg_color="#808080")
        self.back_btn.pack(padx=10, pady=10, ipadx=5, ipady=5, side="bottom")

        self.register_frame = Register(master=self, width=500, height=400, fg_color="transparent", cmd=self.cmd)
        self.register_frame.pack(padx=10, pady=10, expand=True)
        self.register_frame.pack_propagate(0)

    def back_home(self):
        self.back_btn.pack_forget()
        self.or_label.pack_forget()
        self.unlock_btn.pack_forget()
        self.register_frame.pack_forget()

        self.start_btn = customtkinter.CTkButton(master=self, text="Đăng ký hoặc xóa data học sinh", font=self.text_font, command=self.register_user)
        self.start_btn.pack(padx=10, pady=20, ipadx=10, ipady=10)

        self.or_label = customtkinter.CTkLabel(master=self, text="Hoặc", font=self.subtitle_font, fg_color="transparent", text_color="#3E70B5")
        self.or_label.pack(padx=10, pady=0)

        self.unlock_btn = customtkinter.CTkButton(master=self, text="Điểm danh học sinh", font=self.text_font, command=self.test_model)
        self.unlock_btn.pack(padx=10, pady=20, ipadx=10, ipady=10)

    def test_model(self):
        os.system('inv checkface')


class App(customtkinter.CTk):
    def __init__(self, cmd=None):
        super().__init__()

        self.cmd = cmd

        self.title("Hệ thống điểm danh khuôn mặt")
        self.geometry("750x550")

        self.home_frame = Home(master=self, width=700, height=500, fg_color="transparent", cmd=cmd)
        self.home_frame.pack(padx=10, pady=10, expand=True)
