/********************************************************************************
** Form generated from reading UI file 'DBNNRenderController.ui'
**
** Created by: Qt User Interface Compiler version 5.12.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DBNNRENDERCONTROLLER_H
#define UI_DBNNRENDERCONTROLLER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DBNNRenderController
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout_3;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QDoubleSpinBox *doubleSpinBoxDelta;
    QDoubleSpinBox *doubleSpinBoxLookSpeed;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBoxLinearSpeed;
    QLabel *label_5;
    QPushButton *pushButtonStart;
    QPushButton *pushButtonStop;
    QPushButton *pushButtonReset;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_4;
    QScrollBar *scrollbarModels;
    QPushButton *pushButtonSaveView;
    QPushButton *pushButtonSaveViewBatch;
    QSpacerItem *verticalSpacer;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *DBNNRenderController)
    {
        if (DBNNRenderController->objectName().isEmpty())
            DBNNRenderController->setObjectName(QString::fromUtf8("DBNNRenderController"));
        DBNNRenderController->resize(417, 353);
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(DBNNRenderController->sizePolicy().hasHeightForWidth());
        DBNNRenderController->setSizePolicy(sizePolicy);
        DBNNRenderController->setMinimumSize(QSize(300, 300));
        centralwidget = new QWidget(DBNNRenderController);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout_3 = new QGridLayout(centralwidget);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy1);
        groupBox->setMinimumSize(QSize(200, 0));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        doubleSpinBoxDelta = new QDoubleSpinBox(groupBox);
        doubleSpinBoxDelta->setObjectName(QString::fromUtf8("doubleSpinBoxDelta"));
        doubleSpinBoxDelta->setDecimals(3);
        doubleSpinBoxDelta->setSingleStep(0.005000000000000);
        doubleSpinBoxDelta->setValue(0.010000000000000);

        gridLayout_2->addWidget(doubleSpinBoxDelta, 0, 1, 1, 1);

        doubleSpinBoxLookSpeed = new QDoubleSpinBox(groupBox);
        doubleSpinBoxLookSpeed->setObjectName(QString::fromUtf8("doubleSpinBoxLookSpeed"));
        doubleSpinBoxLookSpeed->setValue(50.000000000000000);

        gridLayout_2->addWidget(doubleSpinBoxLookSpeed, 1, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        doubleSpinBoxLinearSpeed = new QDoubleSpinBox(groupBox);
        doubleSpinBoxLinearSpeed->setObjectName(QString::fromUtf8("doubleSpinBoxLinearSpeed"));
        doubleSpinBoxLinearSpeed->setValue(1.000000000000000);

        gridLayout_2->addWidget(doubleSpinBoxLinearSpeed, 2, 1, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_2->addWidget(label_5, 2, 0, 1, 1);


        gridLayout->addLayout(gridLayout_2, 0, 0, 1, 1);

        pushButtonStart = new QPushButton(groupBox);
        pushButtonStart->setObjectName(QString::fromUtf8("pushButtonStart"));

        gridLayout->addWidget(pushButtonStart, 1, 0, 1, 1);

        pushButtonStop = new QPushButton(groupBox);
        pushButtonStop->setObjectName(QString::fromUtf8("pushButtonStop"));

        gridLayout->addWidget(pushButtonStop, 2, 0, 1, 1);

        pushButtonReset = new QPushButton(groupBox);
        pushButtonReset->setObjectName(QString::fromUtf8("pushButtonReset"));

        gridLayout->addWidget(pushButtonReset, 3, 0, 1, 1);


        gridLayout_3->addWidget(groupBox, 0, 0, 1, 1);

        groupBox_2 = new QGroupBox(centralwidget);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        sizePolicy.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy);
        groupBox_2->setMinimumSize(QSize(200, 0));
        gridLayout_4 = new QGridLayout(groupBox_2);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        scrollbarModels = new QScrollBar(groupBox_2);
        scrollbarModels->setObjectName(QString::fromUtf8("scrollbarModels"));
        scrollbarModels->setPageStep(1);
        scrollbarModels->setOrientation(Qt::Horizontal);

        gridLayout_4->addWidget(scrollbarModels, 0, 0, 1, 1);

        pushButtonSaveView = new QPushButton(groupBox_2);
        pushButtonSaveView->setObjectName(QString::fromUtf8("pushButtonSaveView"));

        gridLayout_4->addWidget(pushButtonSaveView, 1, 0, 1, 1);

        pushButtonSaveViewBatch = new QPushButton(groupBox_2);
        pushButtonSaveViewBatch->setObjectName(QString::fromUtf8("pushButtonSaveViewBatch"));

        gridLayout_4->addWidget(pushButtonSaveViewBatch, 2, 0, 1, 1);


        gridLayout_3->addWidget(groupBox_2, 1, 0, 1, 1);

        verticalSpacer = new QSpacerItem(17, 268, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer, 2, 0, 1, 1);

        DBNNRenderController->setCentralWidget(centralwidget);
        menubar = new QMenuBar(DBNNRenderController);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 417, 18));
        DBNNRenderController->setMenuBar(menubar);
        statusbar = new QStatusBar(DBNNRenderController);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        DBNNRenderController->setStatusBar(statusbar);

        retranslateUi(DBNNRenderController);

        QMetaObject::connectSlotsByName(DBNNRenderController);
    } // setupUi

    void retranslateUi(QMainWindow *DBNNRenderController)
    {
        DBNNRenderController->setWindowTitle(QApplication::translate("DBNNRenderController", "MainWindow", nullptr));
        groupBox->setTitle(QApplication::translate("DBNNRenderController", "Camera Control", nullptr));
        label->setText(QApplication::translate("DBNNRenderController", "Delta", nullptr));
        label_4->setText(QApplication::translate("DBNNRenderController", "LookSpeed", nullptr));
        label_5->setText(QApplication::translate("DBNNRenderController", "LinearSpeed", nullptr));
        pushButtonStart->setText(QApplication::translate("DBNNRenderController", "Start", nullptr));
        pushButtonStop->setText(QApplication::translate("DBNNRenderController", "Stop", nullptr));
        pushButtonReset->setText(QApplication::translate("DBNNRenderController", "Reset", nullptr));
        groupBox_2->setTitle(QApplication::translate("DBNNRenderController", "Model Control", nullptr));
        pushButtonSaveView->setText(QApplication::translate("DBNNRenderController", "Save View", nullptr));
        pushButtonSaveViewBatch->setText(QApplication::translate("DBNNRenderController", "Batch Save View", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DBNNRenderController: public Ui_DBNNRenderController {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DBNNRENDERCONTROLLER_H
