/*
source /etc/environment

arm-linux-gnueabihf-g++ webcam_face_pose.cpp -o webcam_face_pose -O3 -std=c++11 \
-I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
-I /usr/local/arm-dlib/install/include/ \
-L /usr/local/arm-dlib/install/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/qt5.5_env/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/usr/lib/ \
-I /usr/local/arm-opencv/install/include/ -L /usr/local/arm-opencv/install/lib/  \
-lpthread -ldlib -lopencv_world

*/

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <time.h>
#include <fcntl.h> 
#include <fstream>
#include <linux/fb.h>
#include <sys/ioctl.h>

using namespace dlib;
using namespace std;


#include <termios.h>
/* reads from keypress, doesn't echo */
int getch(void)
{
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}

bool kbhit()
{
    termios term;
    tcgetattr(0, &term);

    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);

    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);

    tcsetattr(0, TCSANOW, &term);

    return byteswaiting > 0;
}

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // framebuffer depth
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

int main()
{
    try
    {
        cv::VideoCapture cap(2);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
        

        // image_window win;

        // Load face detection and pose estimation models.
        // The first thing we are going to do is load all our models.  First, since we need to
        // find faces in the image we will need a face detector:
        frontal_face_detector detector = get_frontal_face_detector();
        // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
        shape_predictor sp;
        deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("metric_network_renset.dat") >> net;


        std::vector<matrix<rgb_pixel>> faces;


        {
            cv::Mat temp = cv::imread("tsai.png");
       
            cv_image<bgr_pixel> cimg(temp);

            for (auto face : detector(cimg))
            {
                auto shape = sp(cimg, face);

                matrix<rgb_pixel> face_chip;
                extract_image_chip(cimg, get_face_chip_details(shape,125,0.25), face_chip);
                faces.push_back(move(face_chip));
                
            }
        }

        {
            cv::Mat temp = cv::imread("wei.png");
       
            cv_image<bgr_pixel> cimg(temp);


            for (auto face : detector(cimg))
            {
                auto shape = sp(cimg, face);

                matrix<rgb_pixel> face_chip;
                extract_image_chip(cimg, get_face_chip_details(shape,125,0.25), face_chip);
                faces.push_back(move(face_chip));
                
            }
        }

        std::vector<matrix<float,0,1>> embedded = net(faces);



        cv::Size2f image_size;
    
        framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
        std::ofstream ofs("/dev/fb0");

        // Grab and process frames until the main window is closed by the user.
        while(true)
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }

            if(kbhit())
            {
                int ch = getch();
                if(ch == 99)
                {
                    cv::Mat image_show;
                    temp.copyTo(image_show);
                    cvtColor(image_show, image_show, cv::COLOR_BGR2BGR565);

                    clock_t start = clock();

                    cv_image<bgr_pixel> cimg(temp);

                    std::vector<matrix<rgb_pixel>> faces;

                    std::vector<cv::Point> faces_pos;
                    


                    for (auto face : detector(cimg))
                    {
                        auto shape = sp(cimg, face);

                        cv::Point pos;
                        pos.x = face.left();
                        pos.y = face.top();
                        faces_pos.push_back(pos);

                        cv::Rect face_r(face.left(), face.top(), face.width(), face.height());
                        cv::rectangle(image_show,face_r,cv::Scalar(0,255,0),3);


                        int width = shape.part(0).x() - shape.part(2).x();
                        int padding = 5;
                        cv::Rect landmark_r(shape.part(2).x() - padding, shape.part(2).y() - width/4, width + padding*2, width/2);
                        cv::rectangle(image_show,landmark_r,cv::Scalar(255,0,0),3);


                        matrix<rgb_pixel> face_chip;
                        extract_image_chip(cimg, get_face_chip_details(shape,125,0.25), face_chip);
                        faces.push_back(move(face_chip));
                    }


                    image_size = image_show.size();

                    for (int y = 0; y < image_size.height; y++)
                    {
                        ofs.seekp(y*fb_info.xres_virtual*fb_info.bits_per_pixel/8);
                        
                        ofs.write((const char*)image_show.ptr(y),image_size.width*2);
                    }

                    if (faces.size() == 0)
                    {
                        cout << "No faces found in image!" << endl;
                        continue;
                    }

                    // This call asks the DNN to convert each face image in faces into a 128D vector.
                    // In this 128D vector space, images from the same person will be close to each other
                    // but vectors from different people will be far apart.  So we can use these vectors to
                    // identify if a pair of images are from the same person or from different people.  
                    std::vector<matrix<float,0,1>> face_descriptors = net(faces);


                    // In particular, one simple thing we can do is face clustering.  This next bit of code
                    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
                    // algorithm to identify how many people there are and which faces belong to whom.
                    for (size_t i = 0; i < face_descriptors.size(); ++i)
                    {

                        cout << length(embedded[0]-face_descriptors[i]) << " "<< length(embedded[1]-face_descriptors[i])<<endl;

                        string label = "unknown";
                        if (length(embedded[0]-face_descriptors[i]) < net.loss_details().get_distance_threshold())
                        {
                            label = "tsai";
                            cout << "tsai" << endl;
                        }
                        else if(length(embedded[1]-face_descriptors[i]) < net.loss_details().get_distance_threshold())
                        {
                            label = "wei";
                            cout << "wei" << endl;
                        }
                        else
                            cout << "unknown" << endl;

                        faces_pos[i].y -= 5;
                        cv::putText(image_show, label, faces_pos[i], cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8, 0);
                    }

                    cout << "Execute Time: " << (double)(clock()-start)/CLOCKS_PER_SEC * 1000 << " ms" <<endl;
            

                    for (int y = 0; y < image_size.height; y++)
                    {
                        ofs.seekp(y*fb_info.xres_virtual*fb_info.bits_per_pixel/8);
                        
                        ofs.write((const char*)image_show.ptr(y),image_size.width*2);
                    }

                    dlib::sleep(5000);

              
                }
            }
            

            

            cvtColor(temp, temp, cv::COLOR_BGR2BGR565);

            image_size = temp.size();

            for (int y = 0; y < image_size.height; y++)
            {
                ofs.seekp(y*fb_info.xres_virtual*fb_info.bits_per_pixel/8);
                
                ofs.write((const char*)temp.ptr(y),image_size.width*2);
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------




struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html
    int fd = open(framebuffer_device_path,O_RDONLY);
    ioctl(fd,FBIOGET_VSCREENINFO,&screen_info);

    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt

    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    // fb_info.xres_virtual = ......
    // fb_info.bits_per_pixel = ......

    return fb_info;
};