/* Header for class com_xiaomi_mace_JniMaceUtils */

#ifndef MACE_EXAMPLES_ANDROID_MACELIBRARY_SRC_MAIN_CPP_IMAGE_CLASSIFY_H_
#define MACE_EXAMPLES_ANDROID_MACELIBRARY_SRC_MAIN_CPP_IMAGE_CLASSIFY_H_

#include <iostream>
#include <string>


using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

//autumn scenery : Fallen leaves, yellow leaves
//desert : yellow sand, sand dunes, camel(s)
//forest : forest, green tree(s)
//grassland : grass, green
//mountain : stone, green trees
//lake or ocean : lake, ocean, sea
//others : others
//person : face, skin, body
//sky : blue sky, white clouds
//snow scene : snow, white, snow mountain

enum EN_VIDEO_SCENTTYPE { AUTUMN = 1,
    DESERT,
    FOREST,
    GRASSLAND,
    MOUNTAIN,
    LAKEOROCEAN,
    OTHERS,
    PERSON,
    SKY,
    SNOW,
    SPORTS};


string SceneLabel[] = {"autumn", "desert", "forest", "grassland", "mountain", "lake or ocean", "person", "sky", "snow scene", "others"};

struct _ST_video_resultlabels {
    enum EN_VIDEO_SCENTTYPE label;
    char contentText[128];
    float degree;
};


int fpi_video_mobile_net_create_engine();
int fpi_video_mobile_net_classify(unsigned char * buffer, int length, struct _ST_video_resultlabels *result);
int fpi_video_mobile_net_release_source();

#ifdef __cplusplus
}
#endif
#endif  // MACE_EXAMPLES_ANDROID_MACELIBRARY_SRC_MAIN_CPP_IMAGE_CLASSIFY_H_