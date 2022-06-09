import { ChangeDetectorRef, Component, OnInit, ViewChild } from "@angular/core";
import { ResultModel } from "../model/detectionListModel"
import { MatIconRegistry  } from '@angular/material/icon';
import { MatDialog } from "@angular/material/dialog";
import {DomSanitizer} from '@angular/platform-browser';
import { HttpService } from '../http/http-service.service';
import { MatTable, MatTableDataSource } from "@angular/material/table";
import { MatPaginator } from "@angular/material/paginator";
import { animate, state, style, transition, trigger } from "@angular/animations";
import { MatSort } from "@angular/material/sort";


@Component({
  selector: "app-list",
  templateUrl: "./list.component.html",
  styleUrls: ["./list.component.css"],
  animations: [
    trigger('detailExpand', [
      state('void', style({ height: '0px', minHeight: '0', visibility: 'hidden' })),
      state('*', style({ height: '*', visibility: 'visible' })),
      transition('void <=> *', animate('225ms cubic-bezier(0.4, 0.0, 0.2, 1)')),
    ]),
  ],
})

export class ListComponent implements OnInit{
  displayedColumns: string[] = [
    "#",
    "MAC(sinyal)",
    "Telefon Model Sonucu",
    "Güven ve Sayı",
    "Karar",
    "Son Güncellenme Zamanı",
    "KNN Uzay"
  ];

  title: String;
  detectionMap: Map<String, ResultModel>;
  result: ResultModel;
  
  datasource: MatTableDataSource<ResultModel>;

  @ViewChild(MatPaginator) paginator: MatPaginator;
  @ViewChild(MatSort , {static: true}) sort: MatSort;

  constructor( 
    private httpService: HttpService, 
    private matIconRegistry:MatIconRegistry, private domSanitzer:DomSanitizer, 
    public dialog: MatDialog,
    private _cd: ChangeDetectorRef) {

    this.detectionMap = new Map<String, ResultModel>();

    this.datasource = new MatTableDataSource<ResultModel>();
    
    this.matIconRegistry.addSvgIcon(
      'fail',
      this.domSanitzer.bypassSecurityTrustResourceUrl('assets/icons/fail.svg')
    );
    this.matIconRegistry.addSvgIcon(
      'success',
      this.domSanitzer.bypassSecurityTrustResourceUrl('assets/icons/success.svg')
    );

    this.title = "IoT Donanım Kimliklendirme Tabanlı Erişim Kontrol Sistemi"
  }


    ngOnInit() {
      this.datasource.paginator = this.paginator;

      this.httpService.getRfs().subscribe(
        (response) => {
          this.initialDetectMap(response);

          this.datasource.data = Array.from(this.detectionMap.values());
          this.datasource.paginator = this.paginator;
          this.datasource.sort = this.sort;
        },
        (error) => { console.log(error); });

        
    }

    ngAfterViewInit() {
      this.httpService.getRfUpdates();
        
      this.httpService.returnAsObservable().subscribe(
        (response) => {
          if (response.length > 0) {
            this.detectMapNew(JSON.parse(response + ''));
            
            this.datasource.data = Array.from(this.detectionMap.values());
            this.datasource.paginator = this.paginator;
            this.datasource.sort = this.sort;
            this._cd.detectChanges(); 
          }
        }
      );
    }

    detectMapNew(obj: any) {
      if (obj && Object.keys(obj).length === 0
      && Object.getPrototypeOf(obj) === Object.prototype) {
        return;
      }
      var temp = obj;

      try {
        obj = JSON.parse(obj) as ResultModel;
      } catch(e) {
        obj = temp;
      }

      var result = new ResultModel();

      result.MAC = obj.MAC;
      result.Label = obj.Label;
      result.SignalDecision = obj.SignalDecision;
      result.Confidence = obj.Confidence;
      result.RejectInfo = obj.RejectInfo;
      result.LastUpdateTime = obj.LastUpdateTime;
      result.HistoricalResults = obj.HistoricalResults;
      result.ImagePathApproved = obj.ImagePathApproved;
      result.ImagePathRejected = obj.ImagePathRejected;
      result.ApprovedCount = obj.ApprovedCount as number;
      result.RejectedCount = obj.RejectedCount as number;
      result.LastUpdatedUnix =  Date.now() as number;

      this.detectionMap.set(result.MAC,result);
    }


    initialDetectMap(obj: any) {
      if (obj && Object.keys(obj).length === 0
      && Object.getPrototypeOf(obj) === Object.prototype) {
        return;
      }
      obj = JSON.parse(obj);
      var result = new ResultModel();

      (obj as  Array<ResultModel>).map( (el) => {
        result.MAC = el.MAC;
        result.Label = el.Label;
        result.SignalDecision = el.SignalDecision;
        result.Confidence = el.Confidence;
        result.RejectInfo = el.RejectInfo;
        result.LastUpdateTime = el.LastUpdateTime;
        result.ImagePathApproved = el.ImagePathApproved;
        result.ImagePathRejected = el.ImagePathRejected;
        result.HistoricalResults = el.HistoricalResults;
        result.ApprovedCount = el.ApprovedCount as number;
        result.RejectedCount = el.RejectedCount as number;
        result.LastUpdatedUnix =  Date.now() as number;

        this.detectionMap.set(result.MAC,result);
        result = new ResultModel();
      });
      }


  b64EncodeUnicode(str) {
    return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function(match, p1) {
        return String.fromCharCode(parseInt(p1, 16))
    }))
  }

  readTextFile(file) {
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                return rawFile.responseText;
            }
        }
    }
    rawFile.send(null);
  }

  getImage( ImagePathApproved : string, ImagePathRejected : string) {
    if(ImagePathApproved) {
      let dialogRef = this.dialog.open(OpenImagePopUp);
      var imgsrc = 'data:image/png;base64,' + ImagePathApproved;
      var img = new Image(1,1);
      img.src = imgsrc;
      dialogRef.componentInstance.imageBase64 =  imgsrc;
    }
    else if(ImagePathRejected) {
      let dialogRef = this.dialog.open(OpenImagePopUp);
      var imgsrc = 'data:image/png;base64,' + ImagePathRejected;
      var img = new Image(1,1);
      img.src = imgsrc;
      dialogRef.componentInstance.imageBase64 = imgsrc;
    }
  }

  flushRedis(){
    this.httpService.flushRedis().subscribe(
      (response) => {
        this.detectionMap = new Map<String, ResultModel>();
        this.datasource = new MatTableDataSource<ResultModel>();
        this._cd.detectChanges();
      },
      (error) => { console.log(error); });

  }
}

@Component({
  selector: 'image-popup',
  templateUrl: 'image-popup.html',
})
export class OpenImagePopUp {
  imageBase64: any
}
function imageUploaded() {
  throw new Error("Function not implemented.");
}


