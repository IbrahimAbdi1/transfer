-- Name: Ibrahim Abdi utorid: abdiibra
CREATE TABLE IF NOT EXISTS Person (
    uid INT PRIMARY KEY,
    FirstName VARCHAR(100) NOT NULL,
    LastName VARCHAR(100) NOT NULL,
    Gender VARCHAR(20) NOT NULL,
    BirthDate VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS Address (
    uid INT,
    City VARCHAR(100) NOT NULL,
    Street VARCHAR(100) NOT NULL,
    PostalCode VARCHAR(100) NOT NULL,
    Province VARCHAR(100) NOT NULL,
    CONSTRAINT AddressPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, PostalCode)
);

CREATE TABLE IF NOT EXISTS PhoneNumber(
    uid INT,
    Number VARCHAR(100) NOT NULL UNIQUE,
    ContactType VARCHAR(100) NOT NULL CHECK (ContactType IN ('home','work','mobile')),
    CONSTRAINT PhoneNumberPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, Number)
);

CREATE TABLE IF NOT EXISTS Hospital(
    hospitalName VARCHAR(100) PRIMARY KEY,
    StreetAdress VARCHAR(100) NOT NULL,
    City VARCHAR(100) NOT NULL,
    AnnualBudget INT NOT NULL
);

CREATE TABLE IF NOT EXISTS Department(
    hospitalName VARCHAR(100),
    Name VARCHAR(100) NOT NULL UNIQUE,
    AnnualBudget INT NOT NULL,
    CONSTRAINT NurseDepartment_FK FOREIGN KEY (hospitalName) REFERENCES Hospital(hospitalName),
    PRIMARY KEY (hospitalName, Name)
);

CREATE TABLE IF NOT EXISTS Nurse(
    uid INT,
    YearlySalary INT NOT NULL,
    YearsPracticed INT NOT NULL,
    CONSTRAINT Nurse_Person_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, YearlySalary)
);

CREATE TABLE IF NOT EXISTS Physician(
    uid INT,
    YearsPracticed INT NOT NULL,
    YearlySalary INT NOT NULL,
    MedicalSpecialty VARCHAR(100) NOT NULL,
    DepartmentName VARCHAR(100) NOT NULL,
    CONSTRAINT PhysicianPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    CONSTRAINT DepartmentName_FK FOREIGN KEY (DepartmentName) REFERENCES Department(Name),
    PRIMARY KEY (uid, MedicalSpecialty)
);

CREATE TABLE IF NOT EXISTS Patient(
    uid INT,
    physician_uid INT,
    nurse_uid INT,
    HealthInsurance VARCHAR(100) NOT NULL,
    CONSTRAINT PatientPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    CONSTRAINT PhysicianPatient_FK FOREIGN KEY (physician_uid) REFERENCES Physician(uid),
    CONSTRAINT NursePatient_FK FOREIGN KEY(nurse_uid) REFERENCES Nurse(uid),
    PRIMARY KEY (uid, physician_uid)
);
--this is works
CREATE TABLE IF NOT EXISTS Works(
    hospitalName VARCHAR(100),
    DepartmentName VARCHAR(100),
    nurse_uid INT,
    YearlySalary INT,
    CONSTRAINT Nurse_Department_Department_Fk FOREIGN KEY(hospitalName,DepartmentName) REFERENCES Department(hospitalName, Name),
    CONSTRAINT Nurse_Department_Nurse_fk FOREIGN KEY(nurse_uid,YearlySalary) REFERENCES Nurse(uid,YearlySalary)
);

CREATE TABLE IF NOT EXISTS Arrives(
    patient_uid INT, 
    hospitalName VARCHAR(100),
    CONSTRAINT Patient_Arrived_Fk FOREIGN KEY(patient_uid) REFERENCES Patient(uid),
    CONSTRAINT Arrived_hospitalName_Fk FOREIGN KEY(hospitalName) REFERENCES Hospital(hospitalName),
    PRIMARY KEY(patient_uid,hospitalName)
);


CREATE TABLE IF NOT EXISTS AdmissionsRecord(
    AdmitDate VARCHAR(100) NOT NULL,
    Priority VARCHAR(100) NOT NULL,
    patient_uid INT,
    hospitalName VARCHAR(100),
    CONSTRAINT AdmissionsRecord_arrives_FK FOREIGN KEY(patient_uid,hospitalName) REFERENCES Arrives(patient_uid,hospitalName),
    PRIMARY KEY(patient_uid,hospitalName)
);



CREATE TABLE IF NOT EXISTS GivesDiagnosis (
    patient_uid INT,
    physician_uid INT,
    CONSTRAINT GivesDiagnosis_patient_FK FOREIGN KEY(patient_uid,physician_uid) REFERENCES Patient(uid,physician_uid),
    PRIMARY KEY(patient_uid,physician_uid)
);

CREATE TABLE IF NOT EXISTS Diagnosis (
    Disease VARCHAR(100) NOT NULL,
    Date VARCHAR(100) NOT NULL,
    Prognosis VARCHAR(100) NOT NULL,
    patient_uid INT,
    physician_uid INT,
    CONSTRAINT Diagnosis_patient_FK FOREIGN KEY(patient_uid,physician_uid) REFERENCES GivesDiagnosis(patient_uid,physician_uid)
);



CREATE TABLE IF NOT EXISTS MedicalTest(
    unid INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    fee INT
);

CREATE TABLE IF NOT EXISTS Undergoes(
    Test_unid INT,
    Patient_uid INT,
    TestResults VARCHAR(100) NOT NULL,
    TestDate VARCHAR(100) NOT NULL,
    CONSTRAINT MedicalTest_FK FOREIGN KEY(Test_unid) REFERENCES MedicalTest(unid),
    CONSTRAINT Patient_undergo_FK FOREIGN KEY(Patient_uid) REFERENCES Patient(uid),
    PRIMARY KEY(Patient_uid,Test_unid)
);

CREATE TABLE IF NOT EXISTS Perscribes(
    Patient_uid INT,
    Physician_uid INT,
    Date VARCHAR(100) NOT NULL,
    CONSTRAINT Perscribes_patient_FK FOREIGN KEY(Patient_uid,Physician_uid) REFERENCES Patient(uid,physician_uid),
    PRIMARY KEY(Patient_uid,Physician_uid)
);

CREATE TABLE IF NOT EXISTS Perscription(
    Drug_Code INT NOT NULL,
    Dosage VARCHAR(100) NOT NULL,
    Patient_uid INT,
    Physician_uid INT,
    CONSTRAINT Persciption_patient_FK FOREIGN KEY(Patient_uid,Physician_uid) REFERENCES Perscribes(Patient_uid,Physician_uid),
    PRIMARY KEY(Patient_uid,Physician_uid,Drug_Code)
);

CREATE TABLE IF NOT EXISTS Drug(
    Drug_Code INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    DrugCategory VARCHAR(100) NOT NULL,
    UnitCost INT NOT NULL
);

CREATE TABLE IF NOT EXISTS Perscribed(
    Patient_uid INT,
    Physician_uid INT,
    Drug_Code INT,
    CONSTRAINT Perscrbed_FK FOREIGN KEY(Patient_uid,Physician_uid,Drug_Code) REFERENCES Perscription(Patient_uid,Physician_uid,Drug_Code)
);