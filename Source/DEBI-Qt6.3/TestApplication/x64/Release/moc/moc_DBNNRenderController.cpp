/****************************************************************************
** Meta object code from reading C++ file 'DBNNRenderController.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../DBNNRenderController.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'DBNNRenderController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_muw__DBNNRenderController_t {
    QByteArrayData data[14];
    char stringdata0[190];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_muw__DBNNRenderController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_muw__DBNNRenderController_t qt_meta_stringdata_muw__DBNNRenderController = {
    {
QT_MOC_LITERAL(0, 0, 25), // "muw::DBNNRenderController"
QT_MOC_LITERAL(1, 26, 14), // "startAnimation"
QT_MOC_LITERAL(2, 41, 0), // ""
QT_MOC_LITERAL(3, 42, 13), // "stopAnimation"
QT_MOC_LITERAL(4, 56, 14), // "resetAnimation"
QT_MOC_LITERAL(5, 71, 14), // "setLinearSpeed"
QT_MOC_LITERAL(6, 86, 12), // "aLinearSpeed"
QT_MOC_LITERAL(7, 99, 12), // "setLookSpeed"
QT_MOC_LITERAL(8, 112, 10), // "aLookSpeed"
QT_MOC_LITERAL(9, 123, 11), // "selectModel"
QT_MOC_LITERAL(10, 135, 11), // "aModelIndex"
QT_MOC_LITERAL(11, 147, 10), // "grabScreen"
QT_MOC_LITERAL(12, 158, 10), // "aViewIndex"
QT_MOC_LITERAL(13, 169, 20) // "autoPilotModelScroll"

    },
    "muw::DBNNRenderController\0startAnimation\0"
    "\0stopAnimation\0resetAnimation\0"
    "setLinearSpeed\0aLinearSpeed\0setLookSpeed\0"
    "aLookSpeed\0selectModel\0aModelIndex\0"
    "grabScreen\0aViewIndex\0autoPilotModelScroll"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_muw__DBNNRenderController[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x0a /* Public */,
       3,    0,   60,    2, 0x0a /* Public */,
       4,    0,   61,    2, 0x0a /* Public */,
       5,    1,   62,    2, 0x0a /* Public */,
       7,    1,   65,    2, 0x0a /* Public */,
       9,    1,   68,    2, 0x0a /* Public */,
      11,    1,   71,    2, 0x0a /* Public */,
      11,    0,   74,    2, 0x2a /* Public | MethodCloned */,
      13,    0,   75,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    6,
    QMetaType::Void, QMetaType::Double,    8,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void muw::DBNNRenderController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DBNNRenderController *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->startAnimation(); break;
        case 1: _t->stopAnimation(); break;
        case 2: _t->resetAnimation(); break;
        case 3: _t->setLinearSpeed((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->setLookSpeed((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: _t->selectModel((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->grabScreen((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->grabScreen(); break;
        case 8: _t->autoPilotModelScroll(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject muw::DBNNRenderController::staticMetaObject = { {
    &QMainWindow::staticMetaObject,
    qt_meta_stringdata_muw__DBNNRenderController.data,
    qt_meta_data_muw__DBNNRenderController,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *muw::DBNNRenderController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *muw::DBNNRenderController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_muw__DBNNRenderController.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int muw::DBNNRenderController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
