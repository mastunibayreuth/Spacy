TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += nonlinear_elasticity.cpp

HEADERS += \
    cuboid.hh \
    domain_caches.hh\
    functional.hh \
    nonlinear_elasticity.hh \
    parameters.hh \
    setup.hh  \
    trackingFunctional.hh

    
INCLUDEPATH += /opt/lib_extern_kaskade/gcc-5.4.0/dune-2.4.1/include
INCLUDEPATH += /opt/lib_extern_kaskade/gcc-5.4.0/boost-1.59.0/include
INCLUDEPATH += /opt/lib_extern_kaskade/gcc-5.4.0/itsol-1/include
INCLUDEPATH += /opt/lib_extern_kaskade/gcc-5.4.0/taucs-2.0/include
INCLUDEPATH += /opt/lib_extern_kaskade/gcc-5.4.0/umfpack-5.4.0/include
INCLUDEPATH += /home/bt304332/svn-repos/Kaskade7.3
INCLUDEPATH += /home/bt304332/git-repos/FunG
INCLUDEPATH += /home/bt304332/git-repos/UpdateSpacy

LIBS += /home/bt304332/svn-repos/Kaskade7.3/libs/umfpack_solve.o
LIBS += -L/opt/lib_extern_kaskade/gcc-5.4.0/umfpack-5.4.0/lib -lumfpack -lamd
LIBS += -L/home/bt304332/svn-repos/Kaskade7.3/libs -lkaskade
LIBS += -L/opt/lib_extern_kaskade/gcc-5.4.0/dune-2.4.1/lib -ldunegrid -ldunecommon -ldunegeometry  -ldunealugrid $(OPENGLLIB)  -lugS3 -lugS2 -lugL3 -lugL2 -ldevS -ldevX 
LIBS += -L/opt/lib_extern_kaskade/gcc-5.4.0/boost-1.59.0/lib -Wl,-rpath, $(BOOST)/lib -lboost_signals -lboost_program_options -lboost_system -lboost_timer -lboost_thread -lboost_chrono -lpthread
LIBS += -Bstatic -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lgfortran
 -L/home/bt304332/git-repos/UpdateSpacy/build/Spacy -lspacy

QMAKE_CXXFLAGS = -fmessage-length=0 -funroll-loops -DNDEBUG -DHAVE_LIBAMIRA=0


