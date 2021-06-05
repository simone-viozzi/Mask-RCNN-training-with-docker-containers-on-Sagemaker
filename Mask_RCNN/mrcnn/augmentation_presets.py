from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters.meta import Sometimes
from imgaug.augmenters.size import Scale

class aug_presets():
    """
    Collection of various styles of augmentations
    """

    ######################################################################################################
    #   AUGMENTATIONS PRESETS  ###########################################################################
    ######################################################################################################

    @staticmethod
    def psychedelic(self):
        """[summary]
        First exagerate attempt of augmentation, it isn't reccomanded,
        that would be good for a laugh.

        Returns:
            [type]: [description]
        """        
        psychedelic_ = iaa.Sequential([
                iaa.SomeOf((0,1),[
                        iaa.BlendAlphaFrequencyNoise(
                            foreground=iaa.EdgeDetect(1.0),
                            per_channel=True
                        ),
                        iaa.ElasticTransformation(alpha=50, sigma=5),  # apply water effect (affects segmaps)
                        
                        iaa.ReplaceElementwise(
                            iap.FromLowerResolution(
                                iap.Binomial(0.1), size_px=8
                            ),
                            iap.Normal(128, 0.4*128),
                            per_channel=0.5
                        )
                    ]
                ),
                iaa.PiecewiseAffine(scale=iap.Absolute(iap.Normal(0, 0.1))),
                iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                iaa.Affine(
                    rotate=(-45, 45),
                    mode="edge"
                )  # rotate by -45 to 45 degrees (affects segmaps)

            ], random_order=True)
        return psychedelic_

    @staticmethod
    def preset_1(severity=1.0):
        """[summary]
            Apply a large variety of augmentation, included the most disruptives,
            standard augmentations is applayed in the same way but th heaviest augmentations
            can be dimmered in frequency.

            Strong augmentations are also time consuming.

            Returns:
            [type]: [description]
        """   

        aug = iaa.SomeOf((1, 2), [
                aug_presets.aritmetic_aug().maybe_some(p=0.95, n=(1, 3)),
                aug_presets.geometric_aug().maybe_some(p=0.95, n=(1, 3))
            ],
            random_order=True
        )

        return aug

    ######################################################################################################
    #   AUGMENTATIONS SETS DIVIDED BY TYPE  ##############################################################
    ######################################################################################################
    """
        Notes about sets of augmentations.
        each sets is marked with one number, that identify the heaviness of the augmentation, (1 is lower),
        remember more is heavy more is time consuming during the training.
    """

    class base_aug():
        
        aug_list = []
        aug_lists = {}
        n_aug = 0 # number of augmentors in the class list

        def seq(self, rand = True):
            """
            return: the sequence of selected lists
            """
            return iaa.Sequential(self.aug_list, random_order=rand)

        def some(self, n = 0, rand = True):
            """
            return augmenter that apply a subset of augmentations
            default interval of aplayed augmentations (0, max)
            """
            n = (0, self.n_aug) if n == 0 else \
                n if isinstance(n, tuple) else \
                n if isinstance(n, list) else (0, self.n_aug)

            return iaa.SomeOf(n, self.aug_list, random_order=rand)

        def one(self):
            """
            return augmentor that applay one augmentations
            each times taken from the set
            """
            return iaa.OneOf(self.aug_list)

        def maybe_all(self, p=0.5, rand = True):
            """
            return augmentor that if applayed (with probability p) apply all the augmentations in the given set
            """
            return iaa.Sometimes(p, then_list= 
                            iaa.Sequential(self.aug_list, random_order=rand))

        def maybe_some(self, p=0.5, n=0,rand = True):
            """
            return augmentor that if applayed (with probability p) apply a subset of augmentations
            the default interval of aplayed augmentations (0, max)
            """
            n = (0, self.n_aug) if n == 0 else \
                n if isinstance(n, tuple) else \
                n if isinstance(n, list) else (0, self.n_aug)

            return iaa.Sometimes(p, then_list= 
                            iaa.SomeOf(n, self.aug_list, random_order=rand))

        def maybe_one(self, p=0.5, rand = True):
            """
            return augmentor that if applayed (with probability p) apply one augmentation in the given set
            """
            return iaa.Sometimes(p, then_list= 
                            iaa.OneOf(self.aug_list, random_order=rand))
        

    # ARITMETIC ##########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html

    class aritmetic_aug(base_aug):

        def __init__(self, severity=1.0, sets=[0, 1, 2]):
            
            s = severity
            self.aug_lists = {
                0 : [
                    iaa.OneOf([
                            iaa.Add((int(-50*s), int(50*s)), per_channel=True),
                            iaa.AddElementwise((int(-50*s), int(50*s)), per_channel=True)
                        ]
                    )
                ],
                1 : [ 
                    iaa.OneOf([
                            iaa.AdditiveGaussianNoise(scale=iap.Clip(iap.Poisson((-30*s, 30*s)), 0, 255), per_channel=True),
                            iaa.Multiply((0.6*s, 1.4*s)),
                            iaa.Multiply((0.6*s, 1.4*s), per_channel=True)
                        ]
                    )
                ],
                2 : [
                    iaa.OneOf([
                            iaa.OneOf([
                                    iaa.ImpulseNoise((0.025, 0.1)),
                                    iaa.Dropout(p=iap.Uniform((0.5, 0.7*s), 0.9*s))
                                ]
                            ),
                            iaa.OneOf([
                                    iaa.OneOf([
                                        iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1)),
                                        iaa.WithPolarWarping(iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1))),
                                        ]
                                    ),
                                    iaa.OneOf([
                                        iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1), per_channel=True),
                                        iaa.WithPolarWarping(iaa.CoarseSaltAndPepper((0.025*s, 0.05*s), size_percent=(0.03, 0.1), per_channel=True))
                                        ]
                                    )
                                ]
                            ),
                            iaa.OneOf([
                                    iaa.OneOf([
                                            iaa.Cutout(
                                                nb_iterations=(5, 10),
                                                size = (0.05, 0.1),
                                                fill_mode="gaussian",
                                                fill_per_channel=True,
                                                squared=False
                                            ),
                                            iaa.WithPolarWarping(iaa.Cutout(
                                                nb_iterations=(5, 10),
                                                size = (0.05, 0.1),
                                                fill_mode="gaussian",
                                                fill_per_channel=True,
                                                squared=False
                                            ))
                                        ]
                                    ),
                                    iaa.OneOf([
                                            iaa.Cutout(
                                                nb_iterations=(5, 10),
                                                size=(0.05, 0.1),
                                                cval=(0, 255),
                                                fill_per_channel=0.5,
                                                squared=False
                                            ),
                                            iaa.WithPolarWarping(iaa.Cutout(
                                                nb_iterations=(5, 10),
                                                size=(0.05, 0.1),
                                                cval=(0, 255),
                                                fill_per_channel=0.5,
                                                squared=False
                                            ))
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            }
            
            if not isinstance(sets, list):
                sets = [sets]

            for list_ in sets:
                self.aug_list += self.aug_lists[list_]
            
            self.n_aug = len(self.aug_list)
            
        
    # GEOMETRIC ##########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/geometric.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html

    class geometric_aug(base_aug):
        
        def __init__(self, severity=1.0, sets=[0, 1, 2]):
            
            s = severity
            self.aug_lists = {
                0 : [
                    iaa.SomeOf((1, 2), [
                        iaa.Fliplr(0.8), # horizontaly flip with probability
                        iaa.Flipud(0.8), # vertical flip with probability
                        ],
                        random_order=True
                    )
                ],
                1 : [ 
                    iaa.Sometimes(p=0.85, 
                        then_list=iaa.SomeOf((1, 3), [
                                iaa.ScaleX((0.6, 1.4), mode="constant", cval= 0),
                                iaa.ScaleY((0.6, 1.4), mode="constant", cval= 0),
                                iaa.TranslateX(percent=(-0.2, 0.2), mode="constant", cval= 0),
                                iaa.TranslateY(percent=(-0.2, 0.2), mode="constant", cval= 0),
                                iaa.Rotate((-30, 30), mode="constant", cval= 0),
                                iaa.ShearX((-15, 15), mode="constant", cval= 0),
                                iaa.ShearY((-15, 15), mode="constant", cval= 0)
                            ],
                            random_order=True
                        ),
                        else_list=iaa.SomeOf((4, 7), [
                                iaa.ScaleX((0.6, 1.4), mode="constant", cval= 0),
                                iaa.ScaleY((0.6, 1.4), mode="constant", cval= 0),
                                iaa.TranslateX(percent=(-0.2, 0.2), mode="constant", cval= 0),
                                iaa.TranslateY(percent=(-0.2, 0.2), mode="constant", cval= 0),
                                iaa.Rotate((-30, 30), mode="constant", cval= 0),
                                iaa.ShearX((-15, 15), mode="constant", cval= 0),
                                iaa.ShearY((-15, 15), mode="constant", cval= 0)
                            ],
                            random_order=True
                        )
                    )
                ],
                2 : [
                    iaa.OneOf([
                        iaa.PiecewiseAffine(scale=(0.01, 0.05), mode="constant", cval= 0),
                        iaa.ElasticTransformation(alpha=(2.0, 10.0), sigma=(0.1, 1.0), mode="constant", cval= 0),
                        iaa.PerspectiveTransform(scale=(0.05, 0.20), mode="constant", cval= 0)
                    ])
                ]
            }
            
            if not isinstance(sets, list):
                sets = [sets]

            for list_ in sets:
                self.aug_list += self.aug_lists[list_]
            
            self.n_aug = len(self.aug_list)


    # CONTRAST ###########################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html
    


    # COLOR ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/color.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html


    # BLEND ##############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blend.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blend.html
    class blend_aug(base_aug):

        def __init__(self, severity=1.0, sets=[0, 1, 2, 3]):

            s = severity
            self.aug_lists = {
                0: [
                    iaa.BlendAlphaRegularGrid(
                        nb_rows=(15, 40), 
                        nb_cols=(15, 40),
                        foreground=iaa.MotionBlur(k=int(20*s))
                    )
                ],
                1: [
                    iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(1.0),
                                                 iterations=(1, 3),
                                                 upscale_method='linear',
                                                 size_px_max=(2, 8),
                                                 sigmoid=0.2)
                ],
                2: [
                    iaa.BlendAlphaSimplexNoise(
                                iaa.EdgeDetect(iap.Clip(iap.Absolute(iap.Normal(0.4, 0.18)), 0, 1)),
                                upscale_method="linear",
                        per_channel=True)
                ],
                3: [
                    iaa.BlendAlphaSimplexNoise(
                        iaa.AdditiveGaussianNoise(
                            scale=iap.Clip(iap.Poisson(
                                (0, int(60*s))), 0, 255),
                            per_channel=True),
                        upscale_method="linear",
                        per_channel=True)
                ]
            }

            if not isinstance(sets, list):
                sets = [sets]

            for list_ in sets:
                self.aug_list += self.aug_lists[list_]
            
            self.n_aug = len(self.aug_list)

    # BLUR ###############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/blur.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html


    class blur_aug(base_aug):

        def __init__(self, severity=1.0, sets=[0, 1, 2, 3]):

            s = severity
            self.aug_lists = {
                0: [
                    iaa.GaussianBlur(sigma=(0, 5.0*s)),
                ],
                1: [
                    iaa.OneOf([
                        iaa.AverageBlur(k=((15*s, 25*s), (0, 2))),
                        iaa.AverageBlur(k=((0, 2), (15*s, 25*s))),
                    ])
                ],
                2: [
                    iaa.BilateralBlur(
                            d=(4, int(15*s)), sigma_color=(20, 250), sigma_space=(20, 250))
                ],
                3: [
                    iaa.MotionBlur(k=int(13*s))
                ]
            }

            if not isinstance(sets, list):
                sets = [sets]

            for list_ in sets:
                self.aug_list += self.aug_lists[list_]
            
            self.n_aug = len(self.aug_list)

    # CONVOLUTIONAL ######################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/convolutional.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html


    # POOLING ############################################################################
    # overview: https://imgaug.readthedocs.io/en/latest/source/overview/pooling.html
    # docs: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_pooling.html

    

    ######################################################################################
    #   ???
    ######################################################################################
