/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout_3;
    QWidget *widgetView;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QGridLayout *gridLayout;
    QLabel *label;
    QDoubleSpinBox *doubleSpinBoxDeltaX;
    QLabel *label_2;
    QDoubleSpinBox *doubleSpinBoxDeltaY;
    QLabel *label_3;
    QDoubleSpinBox *doubleSpinBoxDeltaZ;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBoxLookSpeed;
    QLabel *label_5;
    QDoubleSpinBox *doubleSpinBoxLinearSpeed;
    QPushButton *pushButtonStart;
    QPushButton *pushButtonStop;
    QPushButton *pushButtonReset;
    QSpacerItem *verticalSpacer;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1133, 1048);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout_3 = new QGridLayout(centralwidget);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        widgetView = new QWidget(centralwidget);
        widgetView->setObjectName(QString::fromUtf8("widgetView"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widgetView->sizePolicy().hasHeightForWidth());
        widgetView->setSizePolicy(sizePolicy);
        widgetView->setMinimumSize(QSize(1000, 1000));
        widgetView->setSizeIncrement(QSize(1, 1));
        widgetView->setBaseSize(QSize(1000, 1000));

        gridLayout_3->addWidget(widgetView, 0, 0, 1, 1);

        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy1);
        groupBox->setMinimumSize(QSize(100, 0));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        doubleSpinBoxDeltaX = new QDoubleSpinBox(groupBox);
        doubleSpinBoxDeltaX->setObjectName(QString::fromUtf8("doubleSpinBoxDeltaX"));

        gridLayout->addWidget(doubleSpinBoxDeltaX, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        doubleSpinBoxDeltaY = new QDoubleSpinBox(groupBox);
        doubleSpinBoxDeltaY->setObjectName(QString::fromUtf8("doubleSpinBoxDeltaY"));

        gridLayout->addWidget(doubleSpinBoxDeltaY, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        doubleSpinBoxDeltaZ = new QDoubleSpinBox(groupBox);
        doubleSpinBoxDeltaZ->setObjectName(QString::fromUtf8("doubleSpinBoxDeltaZ"));

        gridLayout->addWidget(doubleSpinBoxDeltaZ, 2, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        doubleSpinBoxLookSpeed = new QDoubleSpinBox(groupBox);
        doubleSpinBoxLookSpeed->setObjectName(QString::fromUtf8("doubleSpinBoxLookSpeed"));

        gridLayout->addWidget(doubleSpinBoxLookSpeed, 3, 1, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        doubleSpinBoxLinearSpeed = new QDoubleSpinBox(groupBox);
        doubleSpinBoxLinearSpeed->setObjectName(QString::fromUtf8("doubleSpinBoxLinearSpeed"));

        gridLayout->addWidget(doubleSpinBoxLinearSpeed, 4, 1, 1, 1);


        gridLayout_2->addLayout(gridLayout, 0, 0, 1, 1);

        pushButtonStart = new QPushButton(groupBox);
        pushButtonStart->setObjectName(QString::fromUtf8("pushButtonStart"));

        gridLayout_2->addWidget(pushButtonStart, 1, 0, 1, 1);

        pushButtonStop = new QPushButton(groupBox);
        pushButtonStop->setObjectName(QString::fromUtf8("pushButtonStop"));

        gridLayout_2->addWidget(pushButtonStop, 2, 0, 1, 1);

        pushButtonReset = new QPushButton(groupBox);
        pushButtonReset->setObjectName(QString::fromUtf8("pushButtonReset"));

        gridLayout_2->addWidget(pushButtonReset, 3, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 816, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 4, 0, 1, 1);


        gridLayout_3->addWidget(groupBox, 0, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1133, 18));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "Control", nullptr));
        label->setText(QApplication::translate("MainWindow", "DeltaX", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "DeltaY", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "DeltaZ", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "LookSpeed", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "LinearSpeed", nullptr));
        pushButtonStart->setText(QApplication::translate("MainWindow", "Start", nullptr));
        pushButtonStop->setText(QApplication::translate("MainWindow", "Stop", nullptr));
        pushButtonReset->setText(QApplication::translate("MainWindow", "Reset", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
