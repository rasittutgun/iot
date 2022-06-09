import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
//import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { MatFormFieldModule } from "@angular/material/form-field";
import { MatInputModule} from '@angular/material/input';
import { MatCardModule } from '@angular/material/card';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatExpansionModule } from '@angular/material/expansion';

import { AppComponent } from './app.component';


import { ListComponent, OpenImagePopUp } from './list/list.component';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule, } from '@angular/material/paginator';
import { MatIconModule } from '@angular/material/icon';
import { MatDialogModule } from '@angular/material/dialog';
import { MatMenuModule } from '@angular/material/menu';
import { MatButtonModule } from '@angular/material/button';
import { MatDividerModule } from '@angular/material/divider';
import { HttpClientModule } from '@angular/common/http';
import { CdkDetailRowDirective } from './list/cdk-detail-row.directive';

@NgModule({
  declarations: [
    AppComponent,
    ListComponent,
    OpenImagePopUp,
    CdkDetailRowDirective
  ],
  imports: [
    BrowserModule,
    FormsModule,
    MatFormFieldModule,
    MatTableModule,
    MatPaginatorModule,
    BrowserAnimationsModule,
    MatInputModule,
    MatIconModule,
    MatButtonModule,
    MatExpansionModule,
    MatToolbarModule,
    MatMenuModule,
    HttpClientModule,
    MatCardModule,
    MatDialogModule,
    MatDividerModule
  ],
  entryComponents: [ListComponent, OpenImagePopUp],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
