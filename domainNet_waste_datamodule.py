"""
MIT License

Copyright (c) 2023 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""DOMAIN_NET_WASTE DATAMODULE"""
from typing import Callable, Any
from lightningdata.common import pre_process as preprocessor
from lightningdata.modules.domain_adaptation.domainAdaptation_base import DomainAdaptationDataModule

DATASET_NAME = "domainNet_waste"
AVAILABLE_DOMAINS = ["infograph", "painting", "real", "sketch", "clipart"]


class DomainNetWasteDataModule(DomainAdaptationDataModule):
    def __init__(
            self,
            data_dir: str,
            domain: str,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(root=data_dir, name=DATASET_NAME, domain=domain, *args, **kwargs)
        self.available_domains = AVAILABLE_DOMAINS
        # set remote dataset url
        self.remoteFolder = "https://drive.google.com/drive/folders/1kWTmSuPjalFbtIsikfxDshgUzDLuREXf"

    @staticmethod
    def get_domain_names():
        return AVAILABLE_DOMAINS

    @staticmethod
    def get_dataset_name():
        return DATASET_NAME

    def _default_train_transforms(self) -> Callable:
        trans = preprocessor.image_train(resize_size=self.resize_size, crop_size=self.crop_size)
        return trans

    def _default_test_transforms(self) -> Callable:
        trans = preprocessor.image_test(resize_size=self.resize_size, crop_size=self.crop_size)
        return trans