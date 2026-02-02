from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name : str = "Palkesh"
    age : Optional[int] = None
    

new_student = {"age" : 30}
student = Student(**new_student)
student_2 = Student(age = 30, name = "Palkesh")


print(student_2)
print(type(student))
print(student.name)
print(student.age)